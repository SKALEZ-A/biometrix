import { createHash } from 'crypto';

export class NetworkUtils {
  static isPrivateIP(ip: string): boolean {
    const parts = ip.split('.').map(Number);
    
    if (parts[0] === 10) return true;
    if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) return true;
    if (parts[0] === 192 && parts[1] === 168) return true;
    if (parts[0] === 127) return true;
    
    return false;
  }

  static isValidIPv4(ip: string): boolean {
    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    if (!ipv4Regex.test(ip)) return false;

    const parts = ip.split('.').map(Number);
    return parts.every(part => part >= 0 && part <= 255);
  }

  static isValidIPv6(ip: string): boolean {
    const ipv6Regex = /^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/;
    return ipv6Regex.test(ip);
  }

  static anonymizeIP(ip: string): string {
    const parts = ip.split('.');
    if (parts.length === 4) {
      parts[3] = '0';
      return parts.join('.');
    }
    return createHash('sha256').update(ip).digest('hex').substring(0, 16);
  }

  static getIPRange(ip: string, cidr: number): { start: string; end: string } {
    const parts = ip.split('.').map(Number);
    const mask = ~((1 << (32 - cidr)) - 1);
    
    const ipNum = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
    const startNum = ipNum & mask;
    const endNum = startNum | ~mask;

    const toIP = (num: number) => [
      (num >>> 24) & 255,
      (num >>> 16) & 255,
      (num >>> 8) & 255,
      num & 255
    ].join('.');

    return {
      start: toIP(startNum),
      end: toIP(endNum)
    };
  }

  static parseUserAgent(userAgent: string): {
    browser: string;
    version: string;
    os: string;
    device: string;
  } {
    const browser = this.detectBrowser(userAgent);
    const os = this.detectOS(userAgent);
    const device = this.detectDevice(userAgent);

    return { browser: browser.name, version: browser.version, os, device };
  }

  private static detectBrowser(ua: string): { name: string; version: string } {
    if (ua.includes('Chrome')) return { name: 'Chrome', version: this.extractVersion(ua, 'Chrome') };
    if (ua.includes('Firefox')) return { name: 'Firefox', version: this.extractVersion(ua, 'Firefox') };
    if (ua.includes('Safari')) return { name: 'Safari', version: this.extractVersion(ua, 'Version') };
    if (ua.includes('Edge')) return { name: 'Edge', version: this.extractVersion(ua, 'Edge') };
    return { name: 'Unknown', version: '0' };
  }

  private static detectOS(ua: string): string {
    if (ua.includes('Windows')) return 'Windows';
    if (ua.includes('Mac OS')) return 'macOS';
    if (ua.includes('Linux')) return 'Linux';
    if (ua.includes('Android')) return 'Android';
    if (ua.includes('iOS')) return 'iOS';
    return 'Unknown';
  }

  private static detectDevice(ua: string): string {
    if (ua.includes('Mobile')) return 'Mobile';
    if (ua.includes('Tablet')) return 'Tablet';
    return 'Desktop';
  }

  private static extractVersion(ua: string, browser: string): string {
    const match = ua.match(new RegExp(`${browser}\\/(\\d+\\.\\d+)`));
    return match ? match[1] : '0';
  }
}
