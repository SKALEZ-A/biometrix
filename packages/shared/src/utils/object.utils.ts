export class ObjectUtils {
  public static deepClone<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  }

  public static deepMerge<T extends object>(target: T, ...sources: Partial<T>[]): T {
    if (!sources.length) return target;
    const source = sources.shift();

    if (this.isObject(target) && this.isObject(source)) {
      for (const key in source) {
        if (this.isObject(source[key])) {
          if (!target[key]) Object.assign(target, { [key]: {} });
          this.deepMerge(target[key] as any, source[key] as any);
        } else {
          Object.assign(target, { [key]: source[key] });
        }
      }
    }

    return this.deepMerge(target, ...sources);
  }

  public static isObject(item: any): boolean {
    return item && typeof item === 'object' && !Array.isArray(item);
  }

  public static isEmpty(obj: any): boolean {
    if (obj == null) return true;
    if (Array.isArray(obj) || typeof obj === 'string') return obj.length === 0;
    if (typeof obj === 'object') return Object.keys(obj).length === 0;
    return false;
  }

  public static pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
    const result = {} as Pick<T, K>;
    keys.forEach((key) => {
      if (key in obj) {
        result[key] = obj[key];
      }
    });
    return result;
  }

  public static omit<T extends object, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
    const result = { ...obj };
    keys.forEach((key) => {
      delete result[key];
    });
    return result;
  }

  public static flatten(obj: any, prefix: string = ''): { [key: string]: any } {
    const result: { [key: string]: any } = {};

    for (const key in obj) {
      const newKey = prefix ? `${prefix}.${key}` : key;

      if (this.isObject(obj[key]) && !Array.isArray(obj[key])) {
        Object.assign(result, this.flatten(obj[key], newKey));
      } else {
        result[newKey] = obj[key];
      }
    }

    return result;
  }

  public static unflatten(obj: { [key: string]: any }): any {
    const result: any = {};

    for (const key in obj) {
      const keys = key.split('.');
      keys.reduce((acc, k, i) => {
        if (i === keys.length - 1) {
          acc[k] = obj[key];
        } else {
          acc[k] = acc[k] || {};
        }
        return acc[k];
      }, result);
    }

    return result;
  }

  public static getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((acc, key) => acc?.[key], obj);
  }

  public static setNestedValue(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    const lastKey = keys.pop()!;
    const target = keys.reduce((acc, key) => {
      if (!acc[key]) acc[key] = {};
      return acc[key];
    }, obj);
    target[lastKey] = value;
  }

  public static hasNestedKey(obj: any, path: string): boolean {
    return this.getNestedValue(obj, path) !== undefined;
  }

  public static deleteNestedKey(obj: any, path: string): boolean {
    const keys = path.split('.');
    const lastKey = keys.pop()!;
    const target = keys.reduce((acc, key) => acc?.[key], obj);
    if (target && lastKey in target) {
      delete target[lastKey];
      return true;
    }
    return false;
  }

  public static mapKeys<T extends object>(
    obj: T,
    fn: (key: string, value: any) => string
  ): any {
    const result: any = {};
    for (const key in obj) {
      const newKey = fn(key, obj[key]);
      result[newKey] = obj[key];
    }
    return result;
  }

  public static mapValues<T extends object>(
    obj: T,
    fn: (value: any, key: string) => any
  ): T {
    const result: any = {};
    for (const key in obj) {
      result[key] = fn(obj[key], key);
    }
    return result;
  }

  public static filterKeys<T extends object>(
    obj: T,
    predicate: (key: string, value: any) => boolean
  ): Partial<T> {
    const result: any = {};
    for (const key in obj) {
      if (predicate(key, obj[key])) {
        result[key] = obj[key];
      }
    }
    return result;
  }

  public static invert(obj: { [key: string]: any }): { [key: string]: string } {
    const result: { [key: string]: string } = {};
    for (const key in obj) {
      result[obj[key]] = key;
    }
    return result;
  }

  public static isEqual(obj1: any, obj2: any): boolean {
    return JSON.stringify(obj1) === JSON.stringify(obj2);
  }

  public static diff(obj1: any, obj2: any): any {
    const result: any = {};

    for (const key in obj1) {
      if (!(key in obj2)) {
        result[key] = { old: obj1[key], new: undefined };
      } else if (obj1[key] !== obj2[key]) {
        result[key] = { old: obj1[key], new: obj2[key] };
      }
    }

    for (const key in obj2) {
      if (!(key in obj1)) {
        result[key] = { old: undefined, new: obj2[key] };
      }
    }

    return result;
  }
}
