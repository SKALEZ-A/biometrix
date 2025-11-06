export class AsyncUtils {
  public static async sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  public static async retry<T>(
    fn: () => Promise<T>,
    options: {
      maxAttempts?: number;
      delay?: number;
      backoff?: number;
      onRetry?: (attempt: number, error: Error) => void;
    } = {}
  ): Promise<T> {
    const {
      maxAttempts = 3,
      delay = 1000,
      backoff = 2,
      onRetry = () => {},
    } = options;

    let lastError: Error;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        if (attempt < maxAttempts) {
          onRetry(attempt, lastError);
          await this.sleep(delay * Math.pow(backoff, attempt - 1));
        }
      }
    }

    throw lastError!;
  }

  public static async timeout<T>(
    promise: Promise<T>,
    ms: number,
    errorMessage: string = 'Operation timed out'
  ): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) =>
        setTimeout(() => reject(new Error(errorMessage)), ms)
      ),
    ]);
  }

  public static async parallel<T>(
    tasks: (() => Promise<T>)[],
    concurrency: number = Infinity
  ): Promise<T[]> {
    const results: T[] = [];
    const executing: Promise<void>[] = [];

    for (const [index, task] of tasks.entries()) {
      const promise = task().then((result) => {
        results[index] = result;
      });

      executing.push(promise);

      if (executing.length >= concurrency) {
        await Promise.race(executing);
        executing.splice(
          executing.findIndex((p) => p === promise),
          1
        );
      }
    }

    await Promise.all(executing);
    return results;
  }

  public static async sequential<T>(tasks: (() => Promise<T>)[]): Promise<T[]> {
    const results: T[] = [];
    for (const task of tasks) {
      results.push(await task());
    }
    return results;
  }

  public static async waterfall<T>(
    tasks: ((prev: any) => Promise<T>)[],
    initialValue?: any
  ): Promise<T> {
    let result = initialValue;
    for (const task of tasks) {
      result = await task(result);
    }
    return result;
  }

  public static debounce<T extends (...args: any[]) => any>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => fn(...args), delay);
    };
  }

  public static throttle<T extends (...args: any[]) => any>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let lastCall = 0;
    return (...args: Parameters<T>) => {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        fn(...args);
      }
    };
  }

  public static async queue<T>(
    items: T[],
    processor: (item: T) => Promise<void>,
    concurrency: number = 1
  ): Promise<void> {
    const queue = [...items];
    const workers: Promise<void>[] = [];

    const worker = async () => {
      while (queue.length > 0) {
        const item = queue.shift();
        if (item) {
          await processor(item);
        }
      }
    };

    for (let i = 0; i < concurrency; i++) {
      workers.push(worker());
    }

    await Promise.all(workers);
  }

  public static async mapAsync<T, R>(
    items: T[],
    fn: (item: T, index: number) => Promise<R>
  ): Promise<R[]> {
    return Promise.all(items.map((item, index) => fn(item, index)));
  }

  public static async filterAsync<T>(
    items: T[],
    predicate: (item: T, index: number) => Promise<boolean>
  ): Promise<T[]> {
    const results = await Promise.all(
      items.map(async (item, index) => ({
        item,
        keep: await predicate(item, index),
      }))
    );
    return results.filter((r) => r.keep).map((r) => r.item);
  }

  public static async reduceAsync<T, R>(
    items: T[],
    fn: (acc: R, item: T, index: number) => Promise<R>,
    initialValue: R
  ): Promise<R> {
    let accumulator = initialValue;
    for (let i = 0; i < items.length; i++) {
      accumulator = await fn(accumulator, items[i], i);
    }
    return accumulator;
  }
}
