use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "AXPIndexerDaemon",
    version,
    about = "Native document indexing daemon"
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
    Status {
        #[arg(long)]
        db: PathBuf,
    },
    Scan {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        db: PathBuf,
    },
}
