luxeshare/
├─ Anchor.toml                 # конфиг Anchor (кластер, кошелёк, скрипты)
├─ Cargo.toml                  # корневой Cargo workspace
├─ programs/
│  └─ luxeshare/
│     ├─ Cargo.toml
│     └─ src/
│        └─ lib.rs            # Anchor программа (Rust) — логика токенизации/аренды/доходов
├─ ts/
│  ├─ package.json
│  ├─ tsconfig.json
│  ├─ .env.example
│  ├─ idl/
│  │  └─ luxeshare.json       # IDL, генерится anchor'ом (или положи сюда после build)
│  └─ scripts/
│     ├─ setup.ts             # создаёт devnet mint'ы, ATA; вызывает init_asset
│     ├─ flow.ts              # e2e: mint_shares → stake → update_price → rent → claim
│     └─ keeper-update-price.ts # оффчейн-кипер: берёт цену у ML, вызывает update_price
├─ backend/
│  ├─ requirements.txt
│  └─ app.py                  # FastAPI ML-прайсер (заглушка)
└─ web/
   ├─ package.json
   ├─ next.config.js
   └─ app/
      └─ page.tsx             # минимальный UI (Anchor вызовы из браузера)