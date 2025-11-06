export class StringUtils {
  public static capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  public static camelCase(str: string): string {
    return str
      .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
        return index === 0 ? word.toLowerCase() : word.toUpperCase();
      })
      .replace(/\s+/g, '');
  }

  public static snakeCase(str: string): string {
    return str
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '');
  }

  public static kebabCase(str: string): string {
    return str
      .replace(/([a-z])([A-Z])/g, '$1-$2')
      .replace(/\s+/g, '-')
      .toLowerCase();
  }

  public static truncate(str: string, length: number, suffix: string = '...'): string {
    if (str.length <= length) {
      return str;
    }
    return str.substring(0, length - suffix.length) + suffix;
  }

  public static slugify(str: string): string {
    return str
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_-]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  public static reverse(str: string): string {
    return str.split('').reverse().join('');
  }

  public static isPalindrome(str: string): boolean {
    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleaned === this.reverse(cleaned);
  }

  public static countWords(str: string): number {
    return str.trim().split(/\s+/).length;
  }

  public static countOccurrences(str: string, substring: string): number {
    return (str.match(new RegExp(substring, 'g')) || []).length;
  }

  public static replaceAll(str: string, search: string, replace: string): string {
    return str.split(search).join(replace);
  }

  public static padStart(str: string, length: number, char: string = ' '): string {
    return str.padStart(length, char);
  }

  public static padEnd(str: string, length: number, char: string = ' '): string {
    return str.padEnd(length, char);
  }

  public static removeWhitespace(str: string): string {
    return str.replace(/\s+/g, '');
  }

  public static escapeHtml(str: string): string {
    const htmlEscapes: { [key: string]: string } = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    };
    return str.replace(/[&<>"']/g, (char) => htmlEscapes[char]);
  }

  public static unescapeHtml(str: string): string {
    const htmlUnescapes: { [key: string]: string } = {
      '&amp;': '&',
      '&lt;': '<',
      '&gt;': '>',
      '&quot;': '"',
      '&#39;': "'",
    };
    return str.replace(/&(?:amp|lt|gt|quot|#39);/g, (entity) => htmlUnescapes[entity]);
  }

  public static isEmail(str: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(str);
  }

  public static isUrl(str: string): boolean {
    try {
      new URL(str);
      return true;
    } catch {
      return false;
    }
  }

  public static extractUrls(str: string): string[] {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    return str.match(urlRegex) || [];
  }

  public static extractEmails(str: string): string[] {
    const emailRegex = /[^\s@]+@[^\s@]+\.[^\s@]+/g;
    return str.match(emailRegex) || [];
  }

  public static mask(str: string, visibleChars: number = 4, maskChar: string = '*'): string {
    if (str.length <= visibleChars) {
      return str;
    }
    const visible = str.slice(-visibleChars);
    const masked = maskChar.repeat(str.length - visibleChars);
    return masked + visible;
  }

  public static randomString(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }

  public static levenshteinDistance(str1: string, str2: string): number {
    const matrix: number[][] = [];

    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[str2.length][str1.length];
  }

  public static similarity(str1: string, str2: string): number {
    const distance = this.levenshteinDistance(str1, str2);
    const maxLength = Math.max(str1.length, str2.length);
    return maxLength === 0 ? 1 : 1 - distance / maxLength;
  }
}
