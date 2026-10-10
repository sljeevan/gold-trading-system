//+------------------------------------------------------------------+
//|                                           GoldTraderBridge.mq5   |
//|                        Copyright 2026, Institutional Gold Trader |
//|                                       100% Free Open-Source      |
//+------------------------------------------------------------------+
#property copyright "Institutional Gold Trader MCP"
#property link      "https://github.com/gold-trading-system"
#property version   "1.00"
#property description "Free Local TCP Socket Server Bridge for Python MCP Server"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Inputs
input int    InpServerPort    = 8888;       // Local TCP Port to Listen On
input ulong  InpMagicNumber   = 987001;     // Magic Number for Orders
input ulong  InpMaxSlippage   = 20;         // Max Slippage (points)
input int    InpTimerMs       = 100;        // Socket Poll Interval (ms)

//--- Global Objects
CTrade         m_trade;
CPositionInfo  m_position;
CAccountInfo   m_account;
int            m_server_socket = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Helper: Extract string value from simple JSON                    |
//+------------------------------------------------------------------+
string JsonGetString(const string &json, const string &key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   
   int colon = StringFind(json, ":", pos);
   if(colon < 0) return "";
   
   int quote_start = StringFind(json, "\"", colon);
   if(quote_start < 0) return "";
   
   int quote_end = StringFind(json, "\"", quote_start + 1);
   if(quote_end < 0) return "";
   
   return StringSubstr(json, quote_start + 1, quote_end - quote_start - 1);
}

//+------------------------------------------------------------------+
//| Helper: Extract double value from simple JSON                    |
//+------------------------------------------------------------------+
double JsonGetDouble(const string &json, const string &key, double def_val = 0.0)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return def_val;
   
   int colon = StringFind(json, ":", pos);
   if(colon < 0) return def_val;
   
   int start = colon + 1;
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '\"'))
      start++;
      
   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']' || ch == '\"' || ch == ' ')
         break;
      end++;
   }
   
   string val_str = StringSubstr(json, start, end - start);
   return StringToDouble(val_str);
}

//+------------------------------------------------------------------+
//| Helper: Extract int value from simple JSON                       |
//+------------------------------------------------------------------+
long JsonGetLong(const string &json, const string &key, long def_val = 0)
{
   return (long)JsonGetDouble(json, key, (double)def_val);
}

//+------------------------------------------------------------------+
//| Handle ACCOUNT_INFO Command                                      |
//+------------------------------------------------------------------+
string ProcessAccountInfo()
{
   string broker_server = AccountInfoString(ACCOUNT_SERVER);
   long login_id        = AccountInfoInteger(ACCOUNT_LOGIN);
   double balance       = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity        = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin        = AccountInfoDouble(ACCOUNT_MARGIN);
   double free_margin   = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double margin_level  = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   long leverage        = AccountInfoInteger(ACCOUNT_LEVERAGE);
   string currency      = AccountInfoString(ACCOUNT_CURRENCY);
   long trade_mode_val  = AccountInfoInteger(ACCOUNT_TRADE_MODE);
   
   string trade_mode_str = "DEMO";
   if(trade_mode_val == ACCOUNT_TRADE_MODE_REAL)
      trade_mode_str = "LIVE ⚠️ REAL CAPITAL";

   string response = StringFormat(
      "{\"status\":\"SUCCESS\",\"login\":%I64d,\"server\":\"%s\",\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"margin_level_pct\":%.2f,\"leverage\":%I64d,\"currency\":\"%s\",\"trade_mode\":\"%s\"}",
      login_id, broker_server, balance, equity, margin, free_margin, margin_level, leverage, currency, trade_mode_str
   );
   return response;
}

//+------------------------------------------------------------------+
//| Handle GET_POSITIONS Command                                     |
//+------------------------------------------------------------------+
string ProcessGetPositions(const string &filter_sym)
{
   string pos_json = "[";
   int total = PositionsTotal();
   int count = 0;
   
   for(int i = 0; i < total; i++)
   {
      if(m_position.SelectByIndex(i))
      {
         string sym = m_position.Symbol();
         if(filter_sym != "" && StringCompare(sym, filter_sym, false) != 0)
            continue;
            
         if(count > 0) pos_json += ",";
         
         string side = (m_position.PositionType() == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         string open_time = TimeToString(m_position.Time(), TIME_DATE|TIME_SECONDS);
         
         pos_json += StringFormat(
            "{\"ticket\":\"%I64d\",\"symbol\":\"%s\",\"side\":\"%s\",\"lots\":%.2f,\"open_price\":%.2f,\"current_price\":%.2f,\"sl\":%.2f,\"tp\":%.2f,\"pnl\":%.2f,\"open_time\":\"%s\",\"comment\":\"%s\"}",
            m_position.Ticket(), sym, side, m_position.Volume(), m_position.PriceOpen(), m_position.PriceCurrent(),
            m_position.StopLoss(), m_position.TakeProfit(), m_position.Profit(), open_time, m_position.Comment()
         );
         count++;
      }
   }
   pos_json += "]";
   return StringFormat("{\"status\":\"SUCCESS\",\"count\":%d,\"positions\":%s}", count, pos_json);
}

//+------------------------------------------------------------------+
//| Handle PLACE_ORDER Command                                       |
//+------------------------------------------------------------------+
string ProcessPlaceOrder(const string &request_json)
{
   string symbol   = JsonGetString(request_json, "symbol");
   string action   = JsonGetString(request_json, "action");
   double lots     = JsonGetDouble(request_json, "lots", 0.01);
   double sl       = JsonGetDouble(request_json, "sl", 0.0);
   double tp       = JsonGetDouble(request_json, "tp", 0.0);
   string comment  = JsonGetString(request_json, "comment");
   
   if(symbol == "") symbol = _Symbol;
   if(comment == "") comment = "Gold-Trader-MCP";
   
   // Auto-resolve gold symbol variants on XM (XAUUSD, GOLD, XAUUSD.micro, GOLD.micro, etc.)
   if(!SymbolInfoInteger(symbol, SYMBOL_SELECT))
   {
      string candidates[] = {symbol, "XAUUSD", "GOLD", "XAUUSD.micro", "GOLD.micro", "XAUUSD#", "GOLD#", "XAUUSD."};
      for(int c = 0; c < ArraySize(candidates); c++)
      {
         if(SymbolInfoInteger(candidates[c], SYMBOL_VISIBLE) || SymbolSelect(candidates[c], true))
         {
            symbol = candidates[c];
            break;
         }
      }
   }
   
   bool res = false;
   if(StringCompare(action, "BUY", false) == 0)
   {
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      res = m_trade.Buy(lots, symbol, ask, sl, tp, comment);
   }
   else if(StringCompare(action, "SELL", false) == 0)
   {
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      res = m_trade.Sell(lots, symbol, bid, sl, tp, comment);
   }
   else
   {
      return StringFormat("{\"status\":\"ERROR\",\"message\":\"Invalid action '%s' (Use BUY or SELL)\"}", action);
   }
   
   if(res && m_trade.ResultRetcode() == TRADE_RETCODE_DONE)
   {
      ulong order_ticket = m_trade.ResultOrder();
      double price       = m_trade.ResultPrice();
      return StringFormat(
         "{\"status\":\"SUCCESS\",\"ticket\":\"%I64d\",\"symbol\":\"%s\",\"action\":\"%s\",\"lots\":%.2f,\"price\":%.2f,\"sl\":%.2f,\"tp\":%.2f,\"message\":\"XM MT5 order #%I64d filled at $%.2f\"}",
         order_ticket, symbol, action, lots, price, sl, tp, order_ticket, price
      );
   }
   else
   {
      uint retcode = m_trade.ResultRetcode();
      string retmsg = m_trade.ResultComment();
      return StringFormat("{\"status\":\"REJECTED\",\"retcode\":%u,\"message\":\"XM Order failed: %s\"}", retcode, retmsg);
   }
}

//+------------------------------------------------------------------+
//| Handle MODIFY_POSITION Command                                   |
//+------------------------------------------------------------------+
string ProcessModifyPosition(const string &request_json)
{
   ulong ticket = (ulong)JsonGetLong(request_json, "ticket", 0);
   double sl    = JsonGetDouble(request_json, "sl", 0.0);
   double tp    = JsonGetDouble(request_json, "tp", 0.0);
   
   if(ticket == 0)
      return "{\"status\":\"ERROR\",\"message\":\"Missing position ticket\"}";
      
   if(m_trade.PositionModify(ticket, sl, tp))
      return StringFormat("{\"status\":\"SUCCESS\",\"ticket\":\"%I64d\",\"message\":\"Position #%I64d modified (SL=%.2f, TP=%.2f)\"}", ticket, ticket, sl, tp);
   else
      return StringFormat("{\"status\":\"ERROR\",\"ticket\":\"%I64d\",\"message\":\"Modify failed: %s\"}", ticket, m_trade.ResultComment());
}

//+------------------------------------------------------------------+
//| Handle CLOSE_POSITION Command                                    |
//+------------------------------------------------------------------+
string ProcessClosePosition(const string &request_json)
{
   ulong ticket = (ulong)JsonGetLong(request_json, "ticket", 0);
   if(ticket == 0)
      return "{\"status\":\"ERROR\",\"message\":\"Missing position ticket\"}";
      
   if(!m_position.SelectByTicket(ticket))
      return StringFormat("{\"status\":\"ERROR\",\"message\":\"Position #%I64d not found\"}", ticket);
      
   string sym   = m_position.Symbol();
   double vol   = m_position.Volume();
   
   if(m_trade.PositionClose(ticket))
   {
      double price = m_trade.ResultPrice();
      return StringFormat("{\"status\":\"SUCCESS\",\"ticket\":\"%I64d\",\"symbol\":\"%s\",\"lots\":%.2f,\"price\":%.2f,\"message\":\"Position #%I64d closed at $%.2f\"}", ticket, sym, vol, price, ticket, price);
   }
   else
   {
      return StringFormat("{\"status\":\"ERROR\",\"ticket\":\"%I64d\",\"message\":\"Close failed: %s\"}", ticket, m_trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
//| Route Incoming JSON Request to appropriate Handler               |
//+------------------------------------------------------------------+
string RouteRequest(const string &req)
{
   string command = JsonGetString(req, "command");
   
   if(command == "PING")
      return "{\"status\":\"SUCCESS\",\"message\":\"Pong from XM MT5 Bridge EA\"}";
   else if(command == "ACCOUNT_INFO")
      return ProcessAccountInfo();
   else if(command == "GET_POSITIONS")
      return ProcessGetPositions(JsonGetString(req, "symbol"));
   else if(command == "PLACE_ORDER")
      return ProcessPlaceOrder(req);
   else if(command == "MODIFY_POSITION")
      return ProcessModifyPosition(req);
   else if(command == "CLOSE_POSITION")
      return ProcessClosePosition(req);
   else
      return StringFormat("{\"status\":\"ERROR\",\"message\":\"Unknown command '%s'\"}", command);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   m_trade.SetExpertMagicNumber(InpMagicNumber);
   m_trade.SetDeviationInPoints(InpMaxSlippage);
   m_trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   m_server_socket = SocketCreate();
   if(m_server_socket == INVALID_HANDLE)
   {
      PrintFormat("❌ Failed to create socket. Error: %d", GetLastError());
      return INIT_FAILED;
   }
   
   if(!SocketListen(m_server_socket, (uint)InpServerPort))
   {
      PrintFormat("❌ Failed to bind/listen on port %d. Error: %d", InpServerPort, GetLastError());
      SocketClose(m_server_socket);
      m_server_socket = INVALID_HANDLE;
      return INIT_FAILED;
   }
   
   EventSetMillisecondTimer(InpTimerMs);
   PrintFormat("🚀 GoldTraderBridge Socket Server listening on TCP port %d (Magic: %I64d)", InpServerPort, InpMagicNumber);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(m_server_socket != INVALID_HANDLE)
   {
      SocketClose(m_server_socket);
      m_server_socket = INVALID_HANDLE;
      PrintFormat("🛑 GoldTraderBridge Socket Server stopped on port %d", InpServerPort);
   }
}

//+------------------------------------------------------------------+
//| Timer event handler: Accept & Serve Client Connections           |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(m_server_socket == INVALID_HANDLE) return;
   
   int client_socket = SocketAccept(m_server_socket);
   if(client_socket == INVALID_HANDLE) return;
   
   // Read incoming request
   uchar req_buffer[];
   ArrayResize(req_buffer, 4096);
   int bytes_read = SocketRead(client_socket, req_buffer, 4096, 2000);
   
   if(bytes_read > 0)
   {
      string request_str = CharArrayToString(req_buffer, 0, bytes_read);
      string response_str = RouteRequest(request_str) + "\n";
      
      uchar resp_buffer[];
      StringToCharArray(response_str, resp_buffer, 0, StringLen(response_str));
      SocketSend(client_socket, resp_buffer, ArraySize(resp_buffer) - 1);
   }
   
   SocketClose(client_socket);
}
//+------------------------------------------------------------------+
