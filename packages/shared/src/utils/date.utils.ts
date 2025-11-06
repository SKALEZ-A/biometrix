export class DateUtils {
  public static addDays(date: Date, days: number): Date {
    const result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
  }

  public static addHours(date: Date, hours: number): Date {
    const result = new Date(date);
    result.setHours(result.getHours() + hours);
    return result;
  }

  public static addMinutes(date: Date, minutes: number): Date {
    const result = new Date(date);
    result.setMinutes(result.getMinutes() + minutes);
    return result;
  }

  public static differenceInDays(date1: Date, date2: Date): number {
    const diffTime = Math.abs(date2.getTime() - date1.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }

  public static differenceInHours(date1: Date, date2: Date): number {
    const diffTime = Math.abs(date2.getTime() - date1.getTime());
    return Math.floor(diffTime / (1000 * 60 * 60));
  }

  public static differenceInMinutes(date1: Date, date2: Date): number {
    const diffTime = Math.abs(date2.getTime() - date1.getTime());
    return Math.floor(diffTime / (1000 * 60));
  }

  public static isToday(date: Date): boolean {
    const today = new Date();
    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  }

  public static isYesterday(date: Date): boolean {
    const yesterday = this.addDays(new Date(), -1);
    return (
      date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear()
    );
  }

  public static isFuture(date: Date): boolean {
    return date.getTime() > Date.now();
  }

  public static isPast(date: Date): boolean {
    return date.getTime() < Date.now();
  }

  public static startOfDay(date: Date): Date {
    const result = new Date(date);
    result.setHours(0, 0, 0, 0);
    return result;
  }

  public static endOfDay(date: Date): Date {
    const result = new Date(date);
    result.setHours(23, 59, 59, 999);
    return result;
  }

  public static startOfWeek(date: Date): Date {
    const result = new Date(date);
    const day = result.getDay();
    const diff = result.getDate() - day;
    result.setDate(diff);
    return this.startOfDay(result);
  }

  public static endOfWeek(date: Date): Date {
    const result = new Date(date);
    const day = result.getDay();
    const diff = result.getDate() + (6 - day);
    result.setDate(diff);
    return this.endOfDay(result);
  }

  public static startOfMonth(date: Date): Date {
    const result = new Date(date);
    result.setDate(1);
    return this.startOfDay(result);
  }

  public static endOfMonth(date: Date): Date {
    const result = new Date(date);
    result.setMonth(result.getMonth() + 1, 0);
    return this.endOfDay(result);
  }

  public static formatISO(date: Date): string {
    return date.toISOString();
  }

  public static formatDate(date: Date, format: string = 'YYYY-MM-DD'): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');

    return format
      .replace('YYYY', String(year))
      .replace('MM', month)
      .replace('DD', day)
      .replace('HH', hours)
      .replace('mm', minutes)
      .replace('ss', seconds);
  }

  public static parseISO(dateString: string): Date {
    return new Date(dateString);
  }

  public static isValidDate(date: any): boolean {
    return date instanceof Date && !isNaN(date.getTime());
  }

  public static getTimestamp(): number {
    return Date.now();
  }

  public static fromTimestamp(timestamp: number): Date {
    return new Date(timestamp);
  }
}
