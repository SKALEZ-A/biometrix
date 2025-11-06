import { StreamEvent } from '../processors/stream-processor';

export interface WindowConfig {
  type: 'tumbling' | 'sliding' | 'session';
  size: number;
  slide?: number;
  gap?: number;
}

export interface AggregationResult {
  windowStart: number;
  windowEnd: number;
  count: number;
  sum: number;
  avg: number;
  min: number;
  max: number;
  distinctCount: number;
  percentiles: Record<number, number>;
}

export class WindowAggregator {
  private windows: Map<string, Window>;
  private watermark: number = 0;
  private allowedLateness: number = 60000;

  constructor() {
    this.windows = new Map();
  }

  async aggregate(config: {
    events: StreamEvent[];
    windowConfig: WindowConfig;
    aggregationFields: string[];
  }): Promise<AggregationResult[]> {
    const { events, windowConfig, aggregationFields } = config;
    
    const windows = this.createWindows(events, windowConfig);
    const results: AggregationResult[] = [];

    for (const window of windows) {
      const result = this.computeAggregations(window, aggregationFields);
      results.push(result);
    }

    return results;
  }

  private createWindows(events: StreamEvent[], config: WindowConfig): Window[] {
    switch (config.type) {
      case 'tumbling':
        return this.createTumblingWindows(events, config.size);
      case 'sliding':
        return this.createSlidingWindows(events, config.size, config.slide!);
      case 'session':
        return this.createSessionWindows(events, config.gap!);
      default:
        throw new Error(`Unknown window type: ${config.type}`);
    }
  }

  private createTumblingWindows(events: StreamEvent[], size: number): Window[] {
    const windows: Window[] = [];
    const sortedEvents = events.sort((a, b) => a.timestamp - b.timestamp);
    
    if (sortedEvents.length === 0) return windows;

    let currentWindow: Window = {
      start: Math.floor(sortedEvents[0].timestamp / size) * size,
      end: Math.floor(sortedEvents[0].timestamp / size) * size + size,
      events: []
    };

    for (const event of sortedEvents) {
      if (event.timestamp >= currentWindow.end) {
        windows.push(currentWindow);
        currentWindow = {
          start: Math.floor(event.timestamp / size) * size,
          end: Math.floor(event.timestamp / size) * size + size,
          events: []
        };
      }
      currentWindow.events.push(event);
    }

    if (currentWindow.events.length > 0) {
      windows.push(currentWindow);
    }

    return windows;
  }

  private createSlidingWindows(events: StreamEvent[], size: number, slide: number): Window[] {
    const windows: Window[] = [];
    const sortedEvents = events.sort((a, b) => a.timestamp - b.timestamp);
    
    if (sortedEvents.length === 0) return windows;

    const minTimestamp = sortedEvents[0].timestamp;
    const maxTimestamp = sortedEvents[sortedEvents.length - 1].timestamp;

    for (let start = minTimestamp; start <= maxTimestamp; start += slide) {
      const end = start + size;
      const windowEvents = sortedEvents.filter(
        e => e.timestamp >= start && e.timestamp < end
      );

      if (windowEvents.length > 0) {
        windows.push({ start, end, events: windowEvents });
      }
    }

    return windows;
  }

  private createSessionWindows(events: StreamEvent[], gap: number): Window[] {
    const windows: Window[] = [];
    const sortedEvents = events.sort((a, b) => a.timestamp - b.timestamp);
    
    if (sortedEvents.length === 0) return windows;

    let currentWindow: Window = {
      start: sortedEvents[0].timestamp,
      end: sortedEvents[0].timestamp,
      events: [sortedEvents[0]]
    };

    for (let i = 1; i < sortedEvents.length; i++) {
      const event = sortedEvents[i];
      const timeSinceLastEvent = event.timestamp - currentWindow.end;

      if (timeSinceLastEvent <= gap) {
        currentWindow.events.push(event);
        currentWindow.end = event.timestamp;
      } else {
        windows.push(currentWindow);
        currentWindow = {
          start: event.timestamp,
          end: event.timestamp,
          events: [event]
        };
      }
    }

    if (currentWindow.events.length > 0) {
      windows.push(currentWindow);
    }

    return windows;
  }

  private computeAggregations(window: Window, fields: string[]): AggregationResult {
    const values = this.extractValues(window.events, fields);
    const sortedValues = values.sort((a, b) => a - b);

    return {
      windowStart: window.start,
      windowEnd: window.end,
      count: values.length,
      sum: values.reduce((a, b) => a + b, 0),
      avg: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0,
      min: values.length > 0 ? Math.min(...values) : 0,
      max: values.length > 0 ? Math.max(...values) : 0,
      distinctCount: new Set(values).size,
      percentiles: this.calculatePercentiles(sortedValues, [25, 50, 75, 90, 95, 99])
    };
  }

  private extractValues(events: StreamEvent[], fields: string[]): number[] {
    const values: number[] = [];

    for (const event of events) {
      for (const field of fields) {
        const value = this.getNestedValue(event.payload, field);
        if (typeof value === 'number') {
          values.push(value);
        }
      }
    }

    return values;
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  private calculatePercentiles(sortedValues: number[], percentiles: number[]): Record<number, number> {
    const result: Record<number, number> = {};

    for (const p of percentiles) {
      const index = Math.ceil((p / 100) * sortedValues.length) - 1;
      result[p] = sortedValues[Math.max(0, index)] || 0;
    }

    return result;
  }

  updateWatermark(timestamp: number): void {
    this.watermark = timestamp;
    this.evictExpiredWindows();
  }

  private evictExpiredWindows(): void {
    const expiredKeys: string[] = [];

    for (const [key, window] of this.windows.entries()) {
      if (window.end < this.watermark - this.allowedLateness) {
        expiredKeys.push(key);
      }
    }

    for (const key of expiredKeys) {
      this.windows.delete(key);
    }
  }
}

interface Window {
  start: number;
  end: number;
  events: StreamEvent[];
}
