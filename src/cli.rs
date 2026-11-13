use clap::Parser;
use tracing::Level;

#[derive(Debug, Parser)]
pub struct Cli {
    /// ERROR < WARN < INFO < DEBUG < TRACE
    #[arg(short, long, default_value_t = Level::INFO)]
    pub log_level: Level,
    // pub epochs: u64,
}
