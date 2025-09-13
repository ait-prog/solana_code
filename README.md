# В крации о бекенде 
Наш бекенд представляет собой полный смарт-контракт Anchor для  «LuxeShare». Он описывает один «актив» (например, катер), чьи «доли» (SPL-токены) можно стейкать, чтобы получать арендные доходы в USDC. Арендатор платит за дни аренды, комиссия протокола вычитается, а остаток распределяется между стейкерами через reward_per_share-механику с высокой точностью.

# Что делает программа

Создаёт PDA-аккаунт Asset с двумя ATA:
vault (USDC) — получает платежи и платит награды;
staking_vault (shares) — хранит застейканные доли.
Выпускает доли (SPL) актива и даёт их стейкать.
При аренде списывает USDC, берёт комиссию протокола, остаток распределяет стейкерам через reward-per-share (RPS).
Обновляет дневную цену: вручную или через Pyth+ML (бленд 70/30).


# Основные инструкции

init_asset(name, base_daily_price_usd, fee_bps) — создаёт Asset, vault, staking_vault. Проверяет, что shares_mint.mint_authority = PDA.

set_price_updater(updater) — назначает адрес для апдейтов цены.

mint_shares(amount) — минт долей. Только authority.

stake(amount) / unstake(amount) — стейк/анстейк долей.

rent(days) — списывает days * current_daily_price_usd из арендатора:

комиссия → protocol_fee_ata,

остаток → RPS для стейкеров.

claim() — выводит накопленные USDC пользователю.

update_price(new_daily_price_usd) — прямой апдейт цены (authority или price_updater).

update_price_with_pyth(ml_daily_price_usd, max_age_sec) — бленд ML и Pyth.
