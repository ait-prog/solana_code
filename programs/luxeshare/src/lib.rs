use anchor_lang::prelude::*;
use anchor_spl::associated_token::AssociatedToken;
use anchor_spl::token_interface::{self as spl_if, Mint, TokenAccount, TokenInterface, MintTo, Transfer};

// -----------------------------------------------------------------------------
// Program ID
// -----------------------------------------------------------------------------

declare_id!("EF9CQ7WfxzUmTpmQxhMv9WFCJoLkyTKniXdDYXaUg5Kh");

// -----------------------------------------------------------------------------
// Program
// -----------------------------------------------------------------------------

#[program]
pub mod luxeshare {
    use super::*;

    /// Initialize an Asset (PDA) with:
    /// - vault (USDC ATA owned by asset PDA)
    /// - staking_vault (shares ATA owned by asset PDA)
    pub fn init_asset(
        ctx: Context<InitAsset>,
        name: String,
        base_daily_price_usd: u64, // 6 decimals
        fee_bps: u16,              // 0..10000
    ) -> Result<()> {
        require!(fee_bps <= 10_000, LuxErr::InvalidInput);

        let asset = &mut ctx.accounts.asset;
        asset.authority = ctx.accounts.authority.key();
        asset.asset_mint = ctx.accounts.asset_mint.key(); // informational NFT
        asset.shares_mint = ctx.accounts.shares_mint.key(); // SPL fractions
        asset.usdc_mint = ctx.accounts.usdc_mint.key();
        asset.vault = ctx.accounts.vault.key();
        asset.staking_vault = ctx.accounts.staking_vault.key();
        asset.total_staked = 0;
        asset.reward_per_share_acc = 0;
        asset.base_daily_price_usd = base_daily_price_usd;
        asset.current_daily_price_usd = base_daily_price_usd;
        asset.fee_bps = fee_bps;
        asset.price_updater = ctx.accounts.authority.key();
        asset.bump = *ctx.bumps.get("asset").unwrap();
        asset.name = fit_str(name, 64);

        // Additional runtime checks (defense in depth)
        // 1) Vault owners are the asset PDA
        require_keys_eq!(ctx.accounts.vault.owner, ctx.accounts.asset.key(), LuxErr::InvalidVault);
        require_keys_eq!(ctx.accounts.staking_vault.owner, ctx.accounts.asset.key(), LuxErr::InvalidVault);

        // 2) Shares mint authority should be the asset PDA (so program can mint via signer seeds)
        // NOTE: token_interface::Mint exposes the header; use try_borrow_data to read authority.
        // We support both Token-2022 and Token-Program via token_interface.
        let shares_mint_ai = ctx.accounts.shares_mint.to_account_info();
        let mint_header = spl_if::Mint::try_deserialize(&mut shares_mint_ai.data.borrow().as_ref())
            .map_err(|_| LuxErr::InvalidMintAuthority)?;
        if let COption::Some(auth) = mint_header.mint_authority {
            require_keys_eq!(auth, ctx.accounts.asset.key(), LuxErr::InvalidMintAuthority);
        } else {
            // If mint_authority is None → immutable mint (not acceptable here)
            return err!(LuxErr::InvalidMintAuthority);
        }

        emit!(AssetInitialized {
            asset: asset.key(),
            authority: asset.authority,
            shares_mint: asset.shares_mint,
            usdc_mint: asset.usdc_mint,
            fee_bps,
            base_daily_price_usd,
        });
        Ok(())
    }

    /// Set who is allowed to call price updates besides authority
    pub fn set_price_updater(ctx: Context<SetPriceUpdater>, updater: Pubkey) -> Result<()> {
        require_keys_eq!(ctx.accounts.asset.authority, ctx.accounts.authority.key(), LuxErr::Unauthorized);
        ctx.accounts.asset.price_updater = updater;
        emit!(PriceUpdaterSet { asset: ctx.accounts.asset.key(), updater });
        Ok(())
    }

    /// Mint additional shares to an account. Only authority.
    pub fn mint_shares(ctx: Context<MintShares>, amount: u64) -> Result<()> {
        require_keys_eq!(ctx.accounts.asset.authority, ctx.accounts.authority.key(), LuxErr::Unauthorized);
        require!(amount > 0, LuxErr::InvalidInput);

        let seeds = asset_seeds!(&ctx.accounts.asset);
        let cpi = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.shares_mint.to_account_info(),
                to: ctx.accounts.to_shares_ata.to_account_info(),
                authority: ctx.accounts.asset.to_account_info(),
            },
        ).with_signer(&[&seeds]);
        spl_if::mint_to(cpi, amount)?;

        emit!(SharesMinted { asset: ctx.accounts.asset.key(), to: ctx.accounts.to_shares_ata.key(), amount });
        Ok(())
    }

    /// Stake share tokens into the pool to earn rental rewards
    pub fn stake(ctx: Context<Stake>, amount: u64) -> Result<()> {
        require!(amount > 0, LuxErr::InvalidInput);
        let asset = &mut ctx.accounts.asset;
        let pos = &mut ctx.accounts.user_position;

        if pos.initialized == 0 {
            pos.initialized = 1;
            pos.owner = ctx.accounts.user.key();
            pos.asset = asset.key();
            pos.shares_staked = 0;
            pos.reward_debt = 0;
            pos.accrued_rewards = 0;
        } else {
            require_keys_eq!(pos.owner, ctx.accounts.user.key(), LuxErr::Unauthorized);
            require_keys_eq!(pos.asset, asset.key(), LuxErr::InvalidInput);
        }

        // accrue pending first
        let pending = pending_rewards(asset, pos)?;
        pos.accrued_rewards = pos.accrued_rewards.checked_add(pending).ok_or(LuxErr::Overflow)?;

        // transfer shares into the staking vault (owned by asset PDA)
        let cpi = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.user_shares_ata.to_account_info(),
                to: ctx.accounts.staking_vault.to_account_info(),
                authority: ctx.accounts.user.to_account_info(),
            },
        );
        spl_if::transfer(cpi, amount)?;

        pos.shares_staked = pos.shares_staked.checked_add(amount).ok_or(LuxErr::Overflow)?;
        asset.total_staked = asset.total_staked.checked_add(amount).ok_or(LuxErr::Overflow)?;
        pos.reward_debt = mul_div(asset.reward_per_share_acc, pos.shares_staked)?;

        emit!(Staked { asset: asset.key(), user: pos.owner, amount });
        Ok(())
    }

    /// Unstake share tokens and update accounting
    pub fn unstake(ctx: Context<Unstake>, amount: u64) -> Result<()> {
        require!(amount > 0, LuxErr::InvalidInput);
        let asset = &mut ctx.accounts.asset;
        let pos = &mut ctx.accounts.user_position;
        require!(pos.shares_staked >= amount, LuxErr::InsufficientStake);

        // accrue pending first
        let pending = pending_rewards(asset, pos)?;
        pos.accrued_rewards = pos.accrued_rewards.checked_add(pending).ok_or(LuxErr::Overflow)?;

        // transfer shares back to the user using the asset PDA as signer
        let seeds = asset_seeds!(asset);
        let cpi = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.staking_vault.to_account_info(),
                to: ctx.accounts.user_shares_ata.to_account_info(),
                authority: ctx.accounts.asset.to_account_info(),
            },
        ).with_signer(&[&seeds]);
        spl_if::transfer(cpi, amount)?;

        pos.shares_staked = pos.shares_staked - amount;
        asset.total_staked = asset.total_staked - amount;
        pos.reward_debt = mul_div(asset.reward_per_share_acc, pos.shares_staked)?;

        emit!(Unstaked { asset: asset.key(), user: pos.owner, amount });
        Ok(())
    }

    /// Rent the underlying asset for `days`, paying USDC.
    /// Protocol fee is skimmed; the rest is distributed to stakers via RPS.
    pub fn rent(ctx: Context<Rent>, days: u16) -> Result<()> {
        require!(days > 0, LuxErr::InvalidInput);
        let asset = &mut ctx.accounts.asset;

        let price = (asset.current_daily_price_usd as u128) * (days as u128); // 6 decimals
        let fee = price * (asset.fee_bps as u128) / 10_000u128;
        let to_rewards = price - fee;

        // pull USDC from renter → vault
        let cpi_pay = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.renter_usdc_ata.to_account_info(),
                to: ctx.accounts.vault.to_account_info(),
                authority: ctx.accounts.renter.to_account_info(),
            },
        );
        spl_if::transfer(cpi_pay, price as u64)?;

        // send protocol fee out of vault to protocol_fee_ata (asset PDA as signer)
        let seeds = asset_seeds!(asset);
        let cpi_fee = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.vault.to_account_info(),
                to: ctx.accounts.protocol_fee_ata.to_account_info(),
                authority: ctx.accounts.asset.to_account_info(),
            },
        ).with_signer(&[&seeds]);
        spl_if::transfer(cpi_fee, fee as u64)?;

        // RPS (reward-per-share) update from remaining amount
        distribute_rewards(asset, to_rewards as u64)?;

        emit!(Rented { asset: asset.key(), renter: ctx.accounts.renter.key(), days, gross_usdc_6: price as u64, fee_usdc_6: fee as u64 });
        Ok(())
    }

    /// Claim accumulated USDC rewards for the user position
    pub fn claim(ctx: Context<Claim>) -> Result<()> {
        let asset = &mut ctx.accounts.asset;
        let pos = &mut ctx.accounts.user_position;

        let pending = pending_rewards(asset, pos)?;
        let amount = (pos.accrued_rewards as u128)
            .checked_add(pending as u128)
            .ok_or(LuxErr::Overflow)? as u64;
        require!(amount > 0, LuxErr::NothingToClaim);

        let seeds = asset_seeds!(asset);
        let cpi = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.vault.to_account_info(),
                to: ctx.accounts.user_usdc_ata.to_account_info(),
                authority: ctx.accounts.asset.to_account_info(),
            },
        ).with_signer(&[&seeds]);
        spl_if::transfer(cpi, amount)?;

        pos.accrued_rewards = 0;
        pos.reward_debt = mul_div(asset.reward_per_share_acc, pos.shares_staked)?;

        emit!(Claimed { asset: asset.key(), user: pos.owner, amount });
        Ok(())
    }

    /// Off-chain updater pushes a pure value (until Pyth/ML wiring is live)
    pub fn update_price(ctx: Context<UpdatePrice>, new_daily_price_usd: u64) -> Result<()> {
        let caller = ctx.accounts.caller.key();
        require!(caller == ctx.accounts.asset.authority || caller == ctx.accounts.asset.price_updater, LuxErr::Unauthorized);
        require!(new_daily_price_usd > 0, LuxErr::InvalidInput);
        ctx.accounts.asset.current_daily_price_usd = new_daily_price_usd;
        emit!(PriceUpdated { asset: ctx.accounts.asset.key(), daily_price_usd: new_daily_price_usd });
        Ok(())
    }

    /// Update price blending ML and Pyth oracle
    pub fn update_price_with_pyth(ctx: Context<UpdatePriceWithPyth>, ml_daily_price_usd: u64, max_age_sec: u64) -> Result<()> {
        let caller = ctx.accounts.caller.key();
        require!(caller == ctx.accounts.asset.authority || caller == ctx.accounts.asset.price_updater, LuxErr::Unauthorized);
        require!(ml_daily_price_usd > 0, LuxErr::InvalidInput);

        let clock = Clock::get()?;
        let feed = pyth_sdk_solana::load_price_feed_from_account_info(&ctx.accounts.pyth_price_feed)?;
        let price = feed.get_price_no_older_than(clock.slot, max_age_sec).ok_or(LuxErr::InvalidInput)?;
        // Normalize to 6 decimals (USD * 1e6)
        let want_decimals = 6i32;
        let expo = price.expo; // negative for decimals
        // value = price.price * 10^(want_decimals + expo)
        // if expo = -8, want 6 → factor = 10^(6 - 8) = 10^-2
        let factor_pow = (want_decimals as i32) + expo; // can be negative
        let pyth_usd_6 = scale_i64_to_u64(price.price, factor_pow)?;

        let ml = ml_daily_price_usd as u128;
        let pyth = pyth_usd_6 as u128;
        let blended = ((ml * 70 + pyth * 30) / 100) as u64;

        ctx.accounts.asset.current_daily_price_usd = blended;
        emit!(PriceUpdated { asset: ctx.accounts.asset.key(), daily_price_usd: blended });
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Account State
// -----------------------------------------------------------------------------

#[account]
pub struct Asset {
    pub authority: Pubkey,
    pub asset_mint: Pubkey,       // informational NFT
    pub shares_mint: Pubkey,      // SPL fractions
    pub usdc_mint: Pubkey,
    pub vault: Pubkey,            // USDC ATA (authority = asset PDA)
    pub staking_vault: Pubkey,    // shares ATA (authority = asset PDA)
    pub total_staked: u64,
    pub reward_per_share_acc: u128, // SCALE=1e12
    pub current_daily_price_usd: u64,
    pub base_daily_price_usd: u64,
    pub fee_bps: u16,
    pub price_updater: Pubkey,
    pub bump: u8,
    pub name: [u8; 64],
}

#[account]
pub struct UserPosition {
    pub initialized: u8,
    pub owner: Pubkey,
    pub asset: Pubkey,
    pub shares_staked: u64,
    pub reward_debt: u128,
    pub accrued_rewards: u64,
}

// -----------------------------------------------------------------------------
// Events (for indexers / analytics)
// -----------------------------------------------------------------------------

#[event]
pub struct AssetInitialized {
    pub asset: Pubkey,
    pub authority: Pubkey,
    pub shares_mint: Pubkey,
    pub usdc_mint: Pubkey,
    pub fee_bps: u16,
    pub base_daily_price_usd: u64,
}

#[event]
pub struct PriceUpdaterSet { pub asset: Pubkey, pub updater: Pubkey }

#[event]
pub struct SharesMinted { pub asset: Pubkey, pub to: Pubkey, pub amount: u64 }

#[event]
pub struct Staked { pub asset: Pubkey, pub user: Pubkey, pub amount: u64 }

#[event]
pub struct Unstaked { pub asset: Pubkey, pub user: Pubkey, pub amount: u64 }

#[event]
pub struct Rented { pub asset: Pubkey, pub renter: Pubkey, pub days: u16, pub gross_usdc_6: u64, pub fee_usdc_6: u64 }

#[event]
pub struct Claimed { pub asset: Pubkey, pub user: Pubkey, pub amount: u64 }

#[event]
pub struct PriceUpdated { pub asset: Pubkey, pub daily_price_usd: u64 }

// -----------------------------------------------------------------------------
// Constants & Utils
// -----------------------------------------------------------------------------

const SCALE: u128 = 1_000_000_000_000; // 1e12, for high-precision RPS

fn fit_str(s: String, n: usize) -> [u8; 64] {
    let mut out = [0u8; 64];
    let b = s.as_bytes();
    let m = b.len().min(n);
    out[..m].copy_from_slice(&b[..m]);
    out
}

fn mul_div(a: u128, b: u64) -> Result<u128> {
    Ok(a
        .checked_mul(b as u128)
        .ok_or(LuxErr::Overflow)?
        / SCALE)
}

fn pending_rewards(asset: &Account<Asset>, pos: &Account<UserPosition>) -> Result<u64> {
    let curr = mul_div(asset.reward_per_share_acc, pos.shares_staked)?;
    let pending = curr.saturating_sub(pos.reward_debt);
    Ok(pending as u64)
}

fn distribute_rewards(asset: &mut Account<Asset>, new_rewards_usdc: u64) -> Result<()> {
    if asset.total_staked == 0 || new_rewards_usdc == 0 { return Ok(()); }
    let delta = (new_rewards_usdc as u128)
        .checked_mul(SCALE)
        .ok_or(LuxErr::Overflow)?
        / (asset.total_staked as u128);
    asset.reward_per_share_acc = asset
        .reward_per_share_acc
        .checked_add(delta)
        .ok_or(LuxErr::Overflow)?;
    Ok(())
}

/// Scale an i64 oracle price to u64 with `10^pow10` factor (pow10 may be negative)
fn scale_i64_to_u64(v: i64, pow10: i32) -> Result<u64> {
    if pow10 >= 0 {
        let f = 10u128.pow(pow10 as u32);
        let x = (v as i128).checked_mul(f as i128).ok_or(LuxErr::Overflow)?;
        require!(x >= 0, LuxErr::InvalidInput);
        Ok(x as u64)
    } else {
        let d = 10u128.pow((-pow10) as u32);
        let num = (v as i128).abs() as u128;
        let x = num / d;
        Ok(x as u64)
    }
}

// -----------------------------------------------------------------------------
// Accounts
// -----------------------------------------------------------------------------

#[derive(Accounts)]
#[instruction(name: String)]
pub struct InitAsset<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    /// CHECK: informational NFT (not enforced in logic)
    pub asset_mint: InterfaceAccount<'info, Mint>,

    /// Shares mint where mint_authority must be the asset PDA
    #[account(mut)]
    pub shares_mint: InterfaceAccount<'info, Mint>,

    pub usdc_mint: InterfaceAccount<'info, Mint>,

    /// The Asset account is a PDA so it can sign CPIs
    #[account(
        init,
        payer = authority,
        space = 8 + 32*6 + 8*4 + 2 + 1 + 64,
        seeds = [b"asset", asset_mint.key().as_ref()],
        bump
    )]
    pub asset: Account<'info, Asset>,

    /// ATA(USDC) authority = asset PDA
    #[account(
      init,
      payer = authority,
      associated_token::mint = usdc_mint,
      associated_token::authority = asset
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    /// ATA(shares) authority = asset PDA
    #[account(
      init,
      payer = authority,
      associated_token::mint = shares_mint,
      associated_token::authority = asset
    )]
    pub staking_vault: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct MintShares<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(mut, has_one = shares_mint)]
    pub asset: Account<'info, Asset>,

    #[account(mut)]
    pub shares_mint: InterfaceAccount<'info, Mint>,

    #[account(
      mut,
      associated_token::mint = shares_mint,
      associated_token::authority = authority
    )]
    pub to_shares_ata: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

#[derive(Accounts)]
pub struct Stake<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut, has_one = shares_mint, has_one = usdc_mint, has_one = staking_vault)]
    pub asset: Account<'info, Asset>,

    pub shares_mint: InterfaceAccount<'info, Mint>,
    pub usdc_mint: InterfaceAccount<'info, Mint>,

    #[account(
      mut,
      associated_token::mint = shares_mint,
      associated_token::authority = user
    )]
    pub user_shares_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(mut)]
    pub staking_vault: InterfaceAccount<'info, TokenAccount>,

    #[account(
      init_if_needed,
      payer = user,
      space = 8 + 1 + 32 + 32 + 8 + 16 + 8,
      seeds = [b"pos", asset.key().as_ref(), user.key().as_ref()],
      bump
    )]
    pub user_position: Account<'info, UserPosition>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Unstake<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut, has_one = staking_vault, has_one = shares_mint)]
    pub asset: Account<'info, Asset>,

    #[account(mut)]
    pub staking_vault: InterfaceAccount<'info, TokenAccount>,

    #[account(
      mut,
      associated_token::mint = asset.shares_mint,
      associated_token::authority = user
    )]
    pub user_shares_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
      mut,
      seeds = [b"pos", asset.key().as_ref(), user.key().as_ref()],
      bump
    )]
    pub user_position: Account<'info, UserPosition>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

#[derive(Accounts)]
pub struct Rent<'info> {
    #[account(mut)]
    pub renter: Signer<'info>,

    #[account(mut, has_one = usdc_mint, has_one = vault)]
    pub asset: Account<'info, Asset>,

    #[account(mut)]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    #[account(
      mut,
      associated_token::mint = asset.usdc_mint,
      associated_token::authority = renter
    )]
    pub renter_usdc_ata: InterfaceAccount<'info, TokenAccount>,

    /// protocol fee destination (any valid USDC ATA)
    #[account(mut)]
    pub protocol_fee_ata: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

#[derive(Accounts)]
pub struct Claim<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut, has_one = vault, has_one = usdc_mint)]
    pub asset: Account<'info, Asset>,

    #[account(mut)]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    #[account(
      mut,
      associated_token::mint = asset.usdc_mint,
      associated_token::authority = user
    )]
    pub user_usdc_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
      mut,
      seeds = [b"pos", asset.key().as_ref(), user.key().as_ref()],
      bump
    )]
    pub user_position: Account<'info, UserPosition>,

    pub token_program: Program<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

#[derive(Accounts)]
pub struct UpdatePrice<'info> {
    pub caller: Signer<'info>,
    #[account(mut)]
    pub asset: Account<'info, Asset>,
}

#[derive(Accounts)]
pub struct UpdatePriceWithPyth<'info> {
    pub caller: Signer<'info>,
    #[account(mut)]
    pub asset: Account<'info, Asset>,
    /// CHECK: Pyth price account; validated at runtime by pyth-sdk
    pub pyth_price_feed: AccountInfo<'info>,
}

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------

#[error_code]
pub enum LuxErr {
    #[msg("Unauthorized")] Unauthorized,
    #[msg("Nothing to claim")] NothingToClaim,
    #[msg("Insufficient stake")] InsufficientStake,
    #[msg("Invalid input")] InvalidInput,
    #[msg("Invalid vault owner")] InvalidVault,
    #[msg("Invalid mint authority")] InvalidMintAuthority,
    #[msg("Arithmetic overflow")] Overflow,
}

// -----------------------------------------------------------------------------
// PDA Seeds
// -----------------------------------------------------------------------------

macro_rules! asset_seeds {
    ($asset:expr) => {
        [b"asset", $asset.asset_mint.as_ref(), &[$asset.bump]]
    };
}

pub(crate) use asset_seeds;

// -----------------------------------------------------------------------------
// Oracle deps
// -----------------------------------------------------------------------------

use pyth_sdk_solana;
