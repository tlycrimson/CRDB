import logging
import asyncio
import random
import discord


logger = logging.getLogger(__name__)

# --- Bot Runner with Auto-Restart ---
async def run_bot_forever(bot, TOKEN):
    """Run bot with automatic restart on failures"""
    backoff = 1.0
    max_backoff = 300.0
    consecutive_failures = 0
    
    while True:
        try:
            logger.info(f"Starting bot (attempt {consecutive_failures + 1})...")
            
            # Close any existing session before starting
            if bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None
            
            await bot.start(TOKEN)
            
            # Reset on successful run (bot stopped normally)
            backoff = 1.0
            consecutive_failures = 0
            logger.info("Bot stopped normally")
            break
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; shutting down...")
            try:
                await bot.close()
            finally:
                break
                
        except discord.errors.HTTPException as e:
            consecutive_failures += 1
            
            # Clean up session
            if bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None
            
            if getattr(e, "status", None) == 429:
                retry_after = 30
                if hasattr(e, "response") and e.response:
                    retry_after = float(e.response.headers.get('Retry-After', 30))
                logger.warning(f"Rate limited. Waiting {retry_after}s before retry.")
                await asyncio.sleep(retry_after)
            else:
                logger.error(f"HTTPException: {e}")
                
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            
            # Clean up session
            if bot.shared_session and not bot.shared_session.closed:
                await bot.shared_session.close()
                bot.shared_session = None

        # Exponential backoff with jitter
        if consecutive_failures > 0:
            sleep_time = min(max_backoff, backoff * (2 ** consecutive_failures))
            sleep_time *= random.uniform(0.8, 1.2)
            logger.info(f"Restarting in {sleep_time:.1f}s...")
            
            # Ensure bot is closed
            try:
                if not bot.is_closed():
                    await bot.close()
            except:
                pass
                
            await asyncio.sleep(sleep_time)
            backoff = sleep_time


