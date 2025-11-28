export class RiskManager {
  private dailyLoss: number = 0;
  private tradeCount: number = 0;
  private lastTrade: Date = new Date(0);

  canTrade(confidence: number): boolean {
    if (confidence < 0.70) return false; // Min Confidence
    if (Date.now() - this.lastTrade.getTime() < 10 * 1000) return false; // Cooldown
    if (this.tradeCount >= 50) return false; // Max Daily Trades
    return true;
  }

  recordTrade(): void {
    this.tradeCount++;
    this.lastTrade = new Date();
  }
}

export const riskManager = new RiskManager();
