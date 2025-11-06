import * as crypto from 'crypto';

export interface DeviceFingerprintComponents {
  userAgent: string;
  screenResolution: string;
  timezone: string;
  language: string;
  platform: string;
  plugins: string[];
  canvasFingerprint: string;
  webglFingerprint: string;
  audioFingerprint: string;
  fonts: string[];
  cpuClass?: string;
  hardwareConcurrency?: number;
  deviceMemory?: number;
  touchSupport: boolean;
  batteryLevel?: number;
  connectionType?: string;
}

export interface DeviceFingerprintResult {
  fingerprintId: string;
  components: DeviceFingerprintComponents;
  confidence: number;
  timestamp: Date;
  hash: string;
}

export class DeviceFingerprintGenerator {
  /**
   * Generate a unique device fingerprint from browser/device characteristics
   */
  static generateFingerprint(components: DeviceFingerprintComponents): DeviceFingerprintResult {
    const fingerprintString = this.createFingerprintString(components);
    const hash = this.hashFingerprint(fingerprintString);
    const confidence = this.calculateConfidence(components);

    return {
      fingerprintId: `fp_${hash.substring(0, 16)}`,
      components,
      confidence,
      timestamp: new Date(),
      hash,
    };
  }

  /**
   * Create a string representation of all fingerprint components
   */
  private static createFingerprintString(components: DeviceFingerprintComponents): string {
    const parts: string[] = [
      components.userAgent,
      components.screenResolution,
      components.timezone,
      components.language,
      components.platform,
      components.plugins.sort().join(','),
      components.canvasFingerprint,
      components.webglFingerprint,
      components.audioFingerprint,
      components.fonts.sort().join(','),
      components.cpuClass || '',
      String(components.hardwareConcurrency || ''),
      String(components.deviceMemory || ''),
      String(components.touchSupport),
      String(components.batteryLevel || ''),
      components.connectionType || '',
    ];

    return parts.join('|');
  }

  /**
   * Hash the fingerprint string using SHA-256
   */
  private static hashFingerprint(fingerprintString: string): string {
    return crypto.createHash('sha256').update(fingerprintString).digest('hex');
  }

  /**
   * Calculate confidence score based on available components
   */
  private static calculateConfidence(components: DeviceFingerprintComponents): number {
    let score = 0;
    const weights = {
      userAgent: 10,
      screenResolution: 8,
      timezone: 5,
      language: 5,
      platform: 8,
      plugins: 10,
      canvasFingerprint: 15,
      webglFingerprint: 15,
      audioFingerprint: 12,
      fonts: 10,
      cpuClass: 3,
      hardwareConcurrency: 4,
      deviceMemory: 4,
      touchSupport: 3,
      batteryLevel: 2,
      connectionType: 2,
    };

    // User Agent
    if (components.userAgent && components.userAgent.length > 0) {
      score += weights.userAgent;
    }

    // Screen Resolution
    if (components.screenResolution && components.screenResolution.length > 0) {
      score += weights.screenResolution;
    }

    // Timezone
    if (components.timezone && components.timezone.length > 0) {
      score += weights.timezone;
    }

    // Language
    if (components.language && components.language.length > 0) {
      score += weights.language;
    }

    // Platform
    if (components.platform && components.platform.length > 0) {
      score += weights.platform;
    }

    // Plugins
    if (components.plugins && components.plugins.length > 0) {
      score += weights.plugins;
    }

    // Canvas Fingerprint
    if (components.canvasFingerprint && components.canvasFingerprint.length > 0) {
      score += weights.canvasFingerprint;
    }

    // WebGL Fingerprint
    if (components.webglFingerprint && components.webglFingerprint.length > 0) {
      score += weights.webglFingerprint;
    }

    // Audio Fingerprint
    if (components.audioFingerprint && components.audioFingerprint.length > 0) {
      score += weights.audioFingerprint;
    }

    // Fonts
    if (components.fonts && components.fonts.length > 0) {
      score += weights.fonts;
    }

    // CPU Class
    if (components.cpuClass) {
      score += weights.cpuClass;
    }

    // Hardware Concurrency
    if (components.hardwareConcurrency !== undefined) {
      score += weights.hardwareConcurrency;
    }

    // Device Memory
    if (components.deviceMemory !== undefined) {
      score += weights.deviceMemory;
    }

    // Touch Support
    score += weights.touchSupport;

    // Battery Level
    if (components.batteryLevel !== undefined) {
      score += weights.batteryLevel;
    }

    // Connection Type
    if (components.connectionType) {
      score += weights.connectionType;
    }

    // Normalize to 0-1 range
    const maxScore = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    return score / maxScore;
  }

  /**
   * Compare two device fingerprints and return similarity score
   */
  static compareFingerprints(
    fingerprint1: DeviceFingerprintComponents,
    fingerprint2: DeviceFingerprintComponents
  ): number {
    let matchScore = 0;
    let totalWeight = 0;

    const comparisons = [
      {
        weight: 10,
        match: fingerprint1.userAgent === fingerprint2.userAgent,
      },
      {
        weight: 8,
        match: fingerprint1.screenResolution === fingerprint2.screenResolution,
      },
      {
        weight: 5,
        match: fingerprint1.timezone === fingerprint2.timezone,
      },
      {
        weight: 5,
        match: fingerprint1.language === fingerprint2.language,
      },
      {
        weight: 8,
        match: fingerprint1.platform === fingerprint2.platform,
      },
      {
        weight: 10,
        match: this.arraysEqual(fingerprint1.plugins, fingerprint2.plugins),
      },
      {
        weight: 15,
        match: fingerprint1.canvasFingerprint === fingerprint2.canvasFingerprint,
      },
      {
        weight: 15,
        match: fingerprint1.webglFingerprint === fingerprint2.webglFingerprint,
      },
      {
        weight: 12,
        match: fingerprint1.audioFingerprint === fingerprint2.audioFingerprint,
      },
      {
        weight: 10,
        match: this.arraysEqual(fingerprint1.fonts, fingerprint2.fonts),
      },
      {
        weight: 3,
        match: fingerprint1.cpuClass === fingerprint2.cpuClass,
      },
      {
        weight: 4,
        match: fingerprint1.hardwareConcurrency === fingerprint2.hardwareConcurrency,
      },
      {
        weight: 4,
        match: fingerprint1.deviceMemory === fingerprint2.deviceMemory,
      },
      {
        weight: 3,
        match: fingerprint1.touchSupport === fingerprint2.touchSupport,
      },
    ];

    comparisons.forEach(({ weight, match }) => {
      totalWeight += weight;
      if (match) {
        matchScore += weight;
      }
    });

    return totalWeight > 0 ? matchScore / totalWeight : 0;
  }

  /**
   * Check if two arrays are equal
   */
  private static arraysEqual(arr1: string[], arr2: string[]): boolean {
    if (arr1.length !== arr2.length) return false;
    const sorted1 = [...arr1].sort();
    const sorted2 = [...arr2].sort();
    return sorted1.every((val, index) => val === sorted2[index]);
  }

  /**
   * Detect if fingerprint shows signs of spoofing/manipulation
   */
  static detectSpoofing(components: DeviceFingerprintComponents): {
    isSpoofed: boolean;
    reasons: string[];
    confidence: number;
  } {
    const reasons: string[] = [];
    let spoofingScore = 0;

    // Check for inconsistencies

    // 1. User Agent vs Platform mismatch
    if (components.userAgent && components.platform) {
      const uaLower = components.userAgent.toLowerCase();
      const platformLower = components.platform.toLowerCase();

      if (
        (uaLower.includes('windows') && !platformLower.includes('win')) ||
        (uaLower.includes('mac') && !platformLower.includes('mac')) ||
        (uaLower.includes('linux') && !platformLower.includes('linux'))
      ) {
        reasons.push('User Agent and Platform mismatch');
        spoofingScore += 0.3;
      }
    }

    // 2. Touch support inconsistency
    if (components.touchSupport && components.userAgent) {
      const uaLower = components.userAgent.toLowerCase();
      const isMobile = uaLower.includes('mobile') || uaLower.includes('android') || uaLower.includes('iphone');

      if (components.touchSupport && !isMobile) {
        // Desktop with touch is possible (touchscreen laptops)
        // But combined with other factors, could indicate spoofing
      } else if (!components.touchSupport && isMobile) {
        reasons.push('Mobile device without touch support');
        spoofingScore += 0.4;
      }
    }

    // 3. Hardware concurrency anomalies
    if (components.hardwareConcurrency !== undefined) {
      if (components.hardwareConcurrency < 1 || components.hardwareConcurrency > 128) {
        reasons.push('Unusual hardware concurrency value');
        spoofingScore += 0.2;
      }
    }

    // 4. Device memory anomalies
    if (components.deviceMemory !== undefined) {
      const validMemorySizes = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
      if (!validMemorySizes.includes(components.deviceMemory)) {
        reasons.push('Unusual device memory value');
        spoofingScore += 0.2;
      }
    }

    // 5. Empty or minimal fingerprint components
    const componentCount = [
      components.userAgent,
      components.screenResolution,
      components.timezone,
      components.language,
      components.platform,
      components.canvasFingerprint,
      components.webglFingerprint,
      components.audioFingerprint,
    ].filter(c => c && c.length > 0).length;

    if (componentCount < 5) {
      reasons.push('Insufficient fingerprint components');
      spoofingScore += 0.3;
    }

    // 6. Canvas fingerprint anomalies
    if (components.canvasFingerprint) {
      // Check if canvas fingerprint is too generic or blocked
      if (
        components.canvasFingerprint.length < 10 ||
        components.canvasFingerprint === 'blocked' ||
        components.canvasFingerprint === 'unavailable'
      ) {
        reasons.push('Canvas fingerprinting blocked or unavailable');
        spoofingScore += 0.25;
      }
    }

    // 7. WebGL fingerprint anomalies
    if (components.webglFingerprint) {
      if (
        components.webglFingerprint.length < 10 ||
        components.webglFingerprint === 'blocked' ||
        components.webglFingerprint === 'unavailable'
      ) {
        reasons.push('WebGL fingerprinting blocked or unavailable');
        spoofingScore += 0.25;
      }
    }

    // 8. Plugin list anomalies
    if (components.plugins.length === 0) {
      reasons.push('No browser plugins detected');
      spoofingScore += 0.15;
    }

    // 9. Font list anomalies
    if (components.fonts.length < 10) {
      reasons.push('Unusually few fonts detected');
      spoofingScore += 0.15;
    }

    const isSpoofed = spoofingScore >= 0.5;
    const confidence = Math.min(1, spoofingScore);

    return {
      isSpoofed,
      reasons,
      confidence,
    };
  }

  /**
   * Calculate device trust score based on fingerprint characteristics
   */
  static calculateTrustScore(
    components: DeviceFingerprintComponents,
    historicalFingerprints: DeviceFingerprintComponents[]
  ): number {
    let trustScore = 1.0;

    // Check for spoofing
    const spoofingResult = this.detectSpoofing(components);
    if (spoofingResult.isSpoofed) {
      trustScore -= spoofingResult.confidence * 0.5;
    }

    // Check consistency with historical fingerprints
    if (historicalFingerprints.length > 0) {
      const similarities = historicalFingerprints.map(historical =>
        this.compareFingerprints(components, historical)
      );

      const maxSimilarity = Math.max(...similarities);

      if (maxSimilarity < 0.7) {
        // New or significantly different device
        trustScore -= 0.3;
      } else if (maxSimilarity >= 0.95) {
        // Very similar to known device
        trustScore += 0.1;
      }
    } else {
      // First time seeing this device
      trustScore -= 0.2;
    }

    // Check fingerprint quality
    const fingerprintQuality = this.calculateConfidence(components);
    if (fingerprintQuality < 0.5) {
      trustScore -= 0.2;
    }

    return Math.max(0, Math.min(1, trustScore));
  }

  /**
   * Generate a stable device ID that persists across sessions
   */
  static generateStableDeviceId(components: DeviceFingerprintComponents): string {
    // Use only stable components that don't change frequently
    const stableComponents = [
      components.platform,
      components.hardwareConcurrency,
      components.deviceMemory,
      components.touchSupport,
      components.canvasFingerprint,
      components.webglFingerprint,
      components.audioFingerprint,
    ];

    const stableString = stableComponents.join('|');
    const hash = crypto.createHash('sha256').update(stableString).digest('hex');

    return `device_${hash.substring(0, 32)}`;
  }
}

export class DeviceFingerprintValidator {
  /**
   * Validate that all required fingerprint components are present
   */
  static validate(components: Partial<DeviceFingerprintComponents>): {
    isValid: boolean;
    missingComponents: string[];
  } {
    const requiredComponents: (keyof DeviceFingerprintComponents)[] = [
      'userAgent',
      'screenResolution',
      'timezone',
      'language',
      'platform',
      'plugins',
      'canvasFingerprint',
      'webglFingerprint',
      'audioFingerprint',
      'fonts',
      'touchSupport',
    ];

    const missingComponents: string[] = [];

    requiredComponents.forEach(component => {
      if (!components[component]) {
        missingComponents.push(component);
      }
    });

    return {
      isValid: missingComponents.length === 0,
      missingComponents,
    };
  }

  /**
   * Sanitize fingerprint components to prevent injection attacks
   */
  static sanitize(components: DeviceFingerprintComponents): DeviceFingerprintComponents {
    return {
      userAgent: this.sanitizeString(components.userAgent, 500),
      screenResolution: this.sanitizeString(components.screenResolution, 50),
      timezone: this.sanitizeString(components.timezone, 100),
      language: this.sanitizeString(components.language, 50),
      platform: this.sanitizeString(components.platform, 100),
      plugins: components.plugins.map(p => this.sanitizeString(p, 200)).slice(0, 100),
      canvasFingerprint: this.sanitizeString(components.canvasFingerprint, 1000),
      webglFingerprint: this.sanitizeString(components.webglFingerprint, 1000),
      audioFingerprint: this.sanitizeString(components.audioFingerprint, 1000),
      fonts: components.fonts.map(f => this.sanitizeString(f, 100)).slice(0, 500),
      cpuClass: components.cpuClass ? this.sanitizeString(components.cpuClass, 50) : undefined,
      hardwareConcurrency: this.sanitizeNumber(components.hardwareConcurrency, 1, 128),
      deviceMemory: this.sanitizeNumber(components.deviceMemory, 0, 64),
      touchSupport: Boolean(components.touchSupport),
      batteryLevel: this.sanitizeNumber(components.batteryLevel, 0, 1),
      connectionType: components.connectionType
        ? this.sanitizeString(components.connectionType, 50)
        : undefined,
    };
  }

  private static sanitizeString(value: string, maxLength: number): string {
    if (!value) return '';
    // Remove potentially dangerous characters
    const sanitized = value.replace(/[<>\"']/g, '');
    return sanitized.substring(0, maxLength);
  }

  private static sanitizeNumber(
    value: number | undefined,
    min: number,
    max: number
  ): number | undefined {
    if (value === undefined) return undefined;
    if (isNaN(value)) return undefined;
    return Math.max(min, Math.min(max, value));
  }
}
