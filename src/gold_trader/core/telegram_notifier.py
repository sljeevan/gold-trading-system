"""
Telegram Notification Engine for the Institutional Gold Trading System.
Dispatches real-time structured alerts for trade entries, exits, PnL reports, and trailing SL updates.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from gold_trader.config import config

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    """
    Sends rich HTML-formatted trade alerts to Telegram.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.enabled = config.TELEGRAM_NOTIFICATIONS_ENABLED if enabled is None else enabled
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send raw text message to configured Telegram chat."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            logger.debug("Telegram notification skipped (not configured or disabled).")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info("✅ Telegram notification sent successfully.")
                return True
            elif resp.status_code == 400 and parse_mode:
                logger.warning("Telegram parse error, retrying without formatting: %s", resp.text)
                payload.pop("parse_mode", None)
                resp2 = requests.post(self.api_url, json=payload, timeout=10)
                if resp2.status_code == 200 and resp2.json().get("ok"):
                    logger.info("✅ Telegram notification sent successfully (plain-text fallback).")
                    return True
            logger.error("❌ Telegram API error (%s): %s", resp.status_code, resp.text)
            return False
        except Exception as exc:
            logger.error("❌ Failed to send Telegram message: %s", exc)
            return False

    def send_trade_entry(
        self,
        order: Dict[str, Any],
        setup: Dict[str, Any],
    ) -> bool:
        """
        Send detailed trade entry notification with R:R, targets, SL, and rationale.
        """
        symbol = str(order.get("symbol", setup.get("symbol", "XAUUSD")))
        side = str(order.get("action", setup.get("action", "BUY"))).upper()
        lots = float(order.get("lots", setup.get("suggested_lot_size", 0.01)))
        price = float(order.get("price", setup.get("entry_price", 0.0)))
        ticket = str(order.get("ticket", "N/A"))
        sl = setup.get("stop_loss", order.get("sl"))
        tp1 = setup.get("target_1", order.get("tp"))
        tp2 = setup.get("target_2")
        rr = setup.get("risk_reward_t1", 1.5)
        score = setup.get("quality_score", 85)
        setup_name = html.escape(str(setup.get("setup_name", "Institutional Confluence")))
        reasons = setup.get("reasons", [])

        side_icon = "🟢" if side == "BUY" else "🔴"
        action_title = f"{side_icon} <b>NEW {side} TRADE EXECUTED</b>"

        msg = (
            f"🏆 <b>GOLD TRADING SYSTEM — TRADE ENTRY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{action_title}\n\n"
            f"📌 <b>Symbol:</b> <code>{symbol}</code>\n"
            f"🎫 <b>Ticket:</b> <code>#{ticket}</code>\n"
            f"📦 <b>Volume:</b> <code>{lots:.2f} lots</code>\n"
            f"💵 <b>Entry Price:</b> <code>${price:.2f}</code>\n\n"
            f"🎯 <b>Take-Profit 1 (1.5R):</b> <code>${tp1 if tp1 else 'Open'}</code>\n"
            f"🎯 <b>Take-Profit 2 (3.0R):</b> <code>${tp2 if tp2 else 'Open'}</code>\n"
            f"🛑 <b>Stop-Loss:</b> <code>${sl if sl else 'Open'}</code>\n"
            f"⚖️ <b>Risk-to-Reward:</b> <code>{rr:.2f}R</code>\n"
            f"⭐ <b>Quality Score:</b> <code>{score}/100</code>\n\n"
            f"🧠 <b>Strategy Setup:</b> {setup_name}\n"
        )

        if reasons:
            msg += f"<b>Rationale:</b>\n"
            for r in reasons[:3]:
                msg += f" • {html.escape(str(r))}\n"

        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>\n"
            f"🤖 <i>Autonomous 24/7 Gold Engine</i>"
        )
        return self.send_message(msg)

    def send_trade_exit(
        self,
        ticket: str,
        symbol: str,
        side: str,
        lots: float,
        entry_price: float,
        close_price: float,
        pnl: float,
        close_reason: str = "Target / Exit Signal",
        account_balance: Optional[float] = None,
    ) -> bool:
        """
        Send detailed trade exit notification with realized Profit/Loss and balance.
        """
        is_profit = pnl >= 0
        pnl_icon = "💰 🟢" if is_profit else "🛑 🔴"
        status_text = "PROFITABLE TRADE" if is_profit else "STOPPED OUT / LOSS"

        price_diff = (close_price - entry_price) if side.upper() == "BUY" else (entry_price - close_price)
        pips = price_diff * 10.0

        msg = (
            f"🏁 <b>GOLD TRADING SYSTEM — TRADE CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_icon} <b>{status_text}</b>\n\n"
            f"📌 <b>Symbol:</b> <code>{symbol}</code> ({side})\n"
            f"🎫 <b>Ticket:</b> <code>#{ticket}</code>\n"
            f"📦 <b>Volume:</b> <code>{lots:.2f} lots</code>\n"
            f"📥 <b>Entry Price:</b> <code>${entry_price:.2f}</code>\n"
            f"📤 <b>Exit Price:</b> <code>${close_price:.2f}</code>\n\n"
            f"💵 <b>Realized PnL:</b> <code>{'+$' if pnl >= 0 else '-$'}{abs(pnl):.2f}</code> ({'+' if pips >= 0 else ''}{pips:.1f} pips)\n"
            f"📝 <b>Exit Reason:</b> {close_reason}\n"
        )

        if account_balance is not None:
            msg += f"💼 <b>Updated Account Balance:</b> <code>${account_balance:,.2f}</code>\n"

        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>\n"
            f"🤖 <i>Autonomous 24/7 Gold Engine</i>"
        )
        return self.send_message(msg)

    def send_trade_update(
        self,
        ticket: str,
        symbol: str,
        update_type: str,
        details: str,
    ) -> bool:
        """Send trade management update (e.g. Trailing SL to Break-Even)."""
        msg = (
            f"🛡️ <b>TRADE MANAGEMENT UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>Symbol:</b> <code>{symbol}</code> (Ticket #{ticket})\n"
            f"⚙️ <b>Action:</b> {update_type}\n"
            f"📝 <b>Details:</b> {details}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        return self.send_message(msg)

    def send_partial_profit_alert(
        self,
        ticket: str,
        symbol: str,
        closed_lots: float,
        remaining_lots: float,
        realized_pnl: float,
        target_price: float,
        new_sl: float,
        account_balance: Optional[float] = None,
    ) -> bool:
        """Send notification when 50% partial profit is taken at Target 1 with SL moved to Break-Even."""
        msg = (
            f"💰 <b>PARTIAL PROFIT SECURED (TARGET 1 REACHED)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Target 1 Hit:</b> <code>${target_price:.2f}</code> (1.5R)\n"
            f"📌 <b>Symbol:</b> <code>{symbol}</code> (Ticket #{ticket})\n"
            f"📦 <b>Closed Volume:</b> <code>{closed_lots:.2f} lots (50%)</code>\n"
            f"💵 <b>Realized Cash Profit:</b> <code>+${realized_pnl:.2f}</code>\n\n"
            f"🛡️ <b>Risk-Free Status:</b> Remaining <code>{remaining_lots:.2f} lots</code> Stop-Loss adjusted to <b>Break-Even (${new_sl:.2f})</b>.\n"
            f"🚀 <b>Runner Target 2:</b> Aiming for 3.0R extension with zero downside risk!\n"
        )
        if account_balance is not None:
            msg += f"💼 <b>Account Balance:</b> <code>${account_balance:,.2f}</code>\n"
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>\n"
            f"🤖 <i>Autonomous 24/7 Gold Engine</i>"
        )
        return self.send_message(msg)

    def send_daily_summary(
        self,
        date_str: str,
        account_info: Dict[str, Any],
        ladder_status: Dict[str, Any],
        today_trades: List[Dict[str, Any]],
        open_positions: List[Dict[str, Any]],
        market_stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send comprehensive end-of-day trading session report to Telegram.
        """
        balance = float(account_info.get("balance", 0.0))
        equity = float(account_info.get("equity", balance))
        free_margin = float(account_info.get("free_margin", 0.0))
        broker_name = str(account_info.get("broker", "XM MT5"))

        current_level = ladder_status.get("current_level", 1)
        next_lots = ladder_status.get("next_lot_size", 0.03)
        target_balance = ladder_status.get("next_level_balance", 26.0)
        ladder_wins = ladder_status.get("wins", 0)
        ladder_losses = ladder_status.get("losses", 0)
        ladder_scratches = ladder_status.get("scratches", 0)
        peak_level = ladder_status.get("peak_level", current_level)

        total_trades = len(today_trades)
        wins = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) > 0)
        losses = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) < 0)
        scratches = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) == 0)
        daily_pnl = sum(float(t.get("pnl", 0.0)) for t in today_trades)
        decided = wins + losses
        win_rate = (wins / decided * 100) if decided > 0 else 0.0

        pnl_sign = "+$" if daily_pnl >= 0 else "-$"
        pnl_badge = "🟢 PROFITABLE" if daily_pnl > 0 else "🔴 NEGATIVE" if daily_pnl < 0 else "⚪ FLAT"

        msg = (
            f"📊 <b>GOLD TRADING SYSTEM — DAILY SESSION REPORT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>Date:</b> <code>{date_str}</code> (Post-NY Session)\n"
            f"⚡ <b>Session Result:</b> {pnl_badge}\n\n"
            f"💼 <b>ACCOUNT OVERVIEW</b>\n"
            f"• <b>Balance:</b> <code>${balance:,.2f}</code>\n"
            f"• <b>Equity:</b> <code>${equity:,.2f}</code>\n"
            f"• <b>Free Margin:</b> <code>${free_margin:,.2f}</code>\n"
            f"• <b>Broker:</b> <code>{broker_name}</code>\n\n"
            f"🪜 <b>30-LEVEL COMPOUNDING LADDER</b>\n"
            f"• <b>Current Stage:</b> <b>Level {current_level} of 30</b>\n"
            f"• <b>Next Target Balance:</b> <code>${target_balance:,.2f}</code>\n"
            f"• <b>Next Order Lot Size:</b> <code>{next_lots:.2f} lots</code>\n"
            f"• <b>Total Progression Wins:</b> {ladder_wins} | <b>Losses:</b> {ladder_losses}\n"
            f"• <b>Peak Level Reached:</b> Level {peak_level}\n\n"
            f"📈 <b>TODAY'S EXECUTION SUMMARY</b>\n"
            f"• <b>Realized PnL:</b> <code>{pnl_sign}{abs(daily_pnl):,.2f}</code>\n"
            f"• <b>Trades Closed:</b> {total_trades}\n"
            f"• <b>Wins:</b> {wins} 🟢 | <b>Losses:</b> {losses} 🔴 | <b>Breakeven:</b> {scratches} ⚪\n"
        )

        if decided > 0:
            msg += f"• <b>Daily Win Rate:</b> <code>{win_rate:.1f}%</code>\n"

        if today_trades:
            msg += "\n<b>Recent Closed Trades Today:</b>\n"
            for t in today_trades[-5:]:
                t_ticket = t.get("ticket", "N/A")
                t_side = t.get("action", "TRADE")
                t_pnl = float(t.get("pnl", 0.0))
                t_pnl_str = f"+${t_pnl:.2f}" if t_pnl >= 0 else f"-${abs(t_pnl):.2f}"
                msg += f" • #{t_ticket} ({t_side}): <code>{t_pnl_str}</code>\n"
        else:
            msg += "• <i>No closed trades today (market ranging or criteria preserved).</i>\n"

        if open_positions:
            msg += f"\n📂 <b>Active Open Positions ({len(open_positions)}):</b>\n"
            for op in open_positions:
                op_ticket = op.get("ticket", "N/A")
                op_side = op.get("side", op.get("action", "BUY"))
                op_lots = float(op.get("lots", op.get("volume", 0.01)))
                op_pnl = float(op.get("pnl", op.get("profit", 0.0)))
                op_pnl_str = f"+${op_pnl:.2f}" if op_pnl >= 0 else f"-${abs(op_pnl):.2f}"
                msg += f" • #{op_ticket} {op_side} {op_lots:.2f}L | Floating PnL: <code>{op_pnl_str}</code>\n"
        else:
            msg += "\n📂 <b>Active Open Positions:</b> None (Flat overnight)\n"

        if market_stats:
            m_high = market_stats.get("high")
            m_low = market_stats.get("low")
            m_close = market_stats.get("close")
            if m_high and m_low:
                msg += (
                    f"\n🌐 <b>Gold Market Context:</b>\n"
                    f"• Day Range: ${m_low:.2f} – ${m_high:.2f} (Close: ${m_close:.2f})\n"
                )

        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>\n"
            f"🤖 <i>Institutional 24/7 Gold Engine</i>"
        )
        return self.send_message(msg)

