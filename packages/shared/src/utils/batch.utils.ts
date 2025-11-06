export class BatchUtils {
  static async processBatch<T, R>(
    items: T[],
    processor: (item: T) => Promise<R>,
    batchSize: number = 10,
    concurrency: number = 5
  ): Promise<R[]> {
    const results: R[] = [];
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const batchResults = await this.processWithConcurrency(batch, processor, concurrency);
      results.push(...batchResults);
    }
    
    return results;
  }

  static async processWithConcurrency<T, R>(
    items: T[],
    processor: (item: T) => Promise<R>,
    concurrency: number
  ): Promise<R[]> {
    const results: R[] = [];
    const executing: Promise<void>[] = [];
    
    for (const item of items) {
      const promise = processor(item).then(result => {
        results.push(result);
      });
      
      executing.push(promise);
      
      if (executing.length >= concurrency) {
        await Promise.race(executing);
        executing.splice(executing.findIndex(p => p === promise), 1);
      }
    }
    
    await Promise.all(executing);
    return results;
  }

  static chunk<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    
    return chunks;
  }

  static async batchInsert<T>(
    items: T[],
    insertFn: (batch: T[]) => Promise<void>,
    batchSize: number = 100
  ): Promise<void> {
    const batches = this.chunk(items, batchSize);
    
    for (const batch of batches) {
      await insertFn(batch);
    }
  }

  static async batchUpdate<T>(
    items: T[],
    updateFn: (batch: T[]) => Promise<void>,
    batchSize: number = 100
  ): Promise<void> {
    const batches = this.chunk(items, batchSize);
    
    for (const batch of batches) {
      await updateFn(batch);
    }
  }
}
