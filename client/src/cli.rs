use clap::{Parser, Subcommand};
use std::path::PathBuf;

fn parse_limit(value: &str) -> Result<usize, String> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| format!("invalid limit '{value}': expected a whole number from 1 to 100"))?;

    if (1..=100).contains(&limit) {
        Ok(limit)
    } else {
        Err(format!(
            "invalid limit '{value}': expected a value from 1 to 100"
        ))
    }
}

#[derive(Parser)]
#[command(
    name = "AXPIndexerClient",
    version,
    about = "Local AXPIndexer search client"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Subcommand)]
pub enum Command {
    Health {
        #[arg(long)]
        db: PathBuf,
    },
    Search {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 20, value_parser = parse_limit)]
        limit: usize,
    },
    Serve {
        #[arg(long)]
        db: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::parse_limit;

    #[test]
    fn limit_accepts_lower_bound() {
        assert_eq!(parse_limit("1"), Ok(1));
    }

    #[test]
    fn limit_accepts_upper_bound() {
        assert_eq!(parse_limit("100"), Ok(100));
    }

    #[test]
    fn limit_rejects_zero() {
        assert!(parse_limit("0").is_err());
    }

    #[test]
    fn limit_rejects_above_upper_bound() {
        assert!(parse_limit("101").is_err());
    }

    #[test]
    fn limit_rejects_non_numeric_input() {
        assert!(parse_limit("many").is_err());
    }
}
