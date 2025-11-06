export class ArrayUtils {
  public static chunk<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  public static unique<T>(array: T[]): T[] {
    return [...new Set(array)];
  }

  public static uniqueBy<T>(array: T[], key: keyof T): T[] {
    const seen = new Set();
    return array.filter((item) => {
      const value = item[key];
      if (seen.has(value)) {
        return false;
      }
      seen.add(value);
      return true;
    });
  }

  public static groupBy<T>(array: T[], key: keyof T): { [key: string]: T[] } {
    return array.reduce((result, item) => {
      const groupKey = String(item[key]);
      if (!result[groupKey]) {
        result[groupKey] = [];
      }
      result[groupKey].push(item);
      return result;
    }, {} as { [key: string]: T[] });
  }

  public static shuffle<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  public static sample<T>(array: T[], count: number = 1): T[] {
    const shuffled = this.shuffle(array);
    return shuffled.slice(0, count);
  }

  public static flatten<T>(array: (T | T[])[]): T[] {
    return array.reduce((flat, item) => {
      return flat.concat(Array.isArray(item) ? this.flatten(item) : item);
    }, [] as T[]);
  }

  public static intersection<T>(array1: T[], array2: T[]): T[] {
    const set2 = new Set(array2);
    return array1.filter((item) => set2.has(item));
  }

  public static difference<T>(array1: T[], array2: T[]): T[] {
    const set2 = new Set(array2);
    return array1.filter((item) => !set2.has(item));
  }

  public static union<T>(...arrays: T[][]): T[] {
    return this.unique(this.flatten(arrays));
  }

  public static partition<T>(array: T[], predicate: (item: T) => boolean): [T[], T[]] {
    const pass: T[] = [];
    const fail: T[] = [];
    array.forEach((item) => {
      if (predicate(item)) {
        pass.push(item);
      } else {
        fail.push(item);
      }
    });
    return [pass, fail];
  }

  public static sortBy<T>(array: T[], key: keyof T, order: 'asc' | 'desc' = 'asc'): T[] {
    return [...array].sort((a, b) => {
      const aVal = a[key];
      const bVal = b[key];
      if (aVal < bVal) return order === 'asc' ? -1 : 1;
      if (aVal > bVal) return order === 'asc' ? 1 : -1;
      return 0;
    });
  }

  public static sum(array: number[]): number {
    return array.reduce((sum, num) => sum + num, 0);
  }

  public static average(array: number[]): number {
    return array.length > 0 ? this.sum(array) / array.length : 0;
  }

  public static median(array: number[]): number {
    const sorted = [...array].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  public static min(array: number[]): number {
    return Math.min(...array);
  }

  public static max(array: number[]): number {
    return Math.max(...array);
  }

  public static range(start: number, end: number, step: number = 1): number[] {
    const result: number[] = [];
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
    return result;
  }

  public static isEmpty<T>(array: T[]): boolean {
    return array.length === 0;
  }

  public static first<T>(array: T[]): T | undefined {
    return array[0];
  }

  public static last<T>(array: T[]): T | undefined {
    return array[array.length - 1];
  }

  public static compact<T>(array: (T | null | undefined)[]): T[] {
    return array.filter((item): item is T => item != null);
  }
}
