import { StreamEvent } from './stream-processor';

export interface WatermarkStrategy {
  type: 'periodic' | 'punctuated';
  interval?: number;
  maxOutOfOrderness?: number;
}

export class EventTimeProcessor {
  private currentWatermark: number = 0;
  private maxOutOfOrderness: number = 5000;
  private watermarkInterval: number = 1000;
  private eventBuffer: StreamEvent[] = [];
  private watermarkCallbacks: Array<(watermark: number) => void> = [];

  constructor(strategy?: WatermarkStrategy) {
    if (strategy) {
      this.maxOutOfOrderness = strategy.maxOutOfOrderness || 5000;
      this.watermarkInterval = strategy.interval || 1000;
    }
  }

  processEvent(event: StreamEvent): void {
    this.eventBuffer.push(event);
    this.eventBuffer.sort((a, b) => a.timestamp - b.timestamp);

    if (this.eventBuffer.length > 1000) {
      this.eventBuffer = this.eventBuffer.slice(-1000);
    }

    this.updateWatermark();
  }

  private updateWatermark(): void {
    if (this.eventBuffer.length === 0) return;

    const maxTimestamp = Math.max(...this.eventBuffer.map(e => e.timestamp));
    const newWatermark = maxTimestamp - this.maxOutOfOrderness;

    if (newWatermark > this.currentWatermark) {
      this.currentWatermark = newWatermark;
      this.notifyWatermarkUpdate(newWatermark);
    }
  }

  private notifyWatermarkUpdate(watermark: number): void {
    for (const callback of this.watermarkCallbacks) {
      callback(watermark);
    }
  }

  onWatermarkUpdate(callback: (watermark: number) => void): void {
    this.watermarkCallbacks.push(callback);
  }

  getCurrentWatermark(): number {
    return this.currentWatermark;
  }

  isEventLate(event: StreamEvent): boolean {
    return event.timestamp < this.currentWatermark;
  }

  getEventTimeDelay(event: StreamEvent): number {
    return Date.now() - event.timestamp;
  }

  getProcessingTimeDelay(event: StreamEvent): number {
    const processingTime = event.metadata?.processingTimestamp || Date.now();
    return processingTime - event.timestamp;
  }
}
