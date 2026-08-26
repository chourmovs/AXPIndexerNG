use clap::{Parser, Subcommand};
use std::path::PathBuf;
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
        #[arg(long,default_value_t=20,value_parser=clap::value_parser!(usize).range(1..=100))]
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
