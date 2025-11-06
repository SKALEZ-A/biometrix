import { Request, Response, NextFunction } from 'express';

export enum VersionStrategy {
  HEADER = 'header',
  URL = 'url',
  QUERY = 'query',
  ACCEPT_HEADER = 'accept-header'
}

interface VersionConfig {
  strategy: VersionStrategy;
  headerName?: string;
  queryParam?: string;
  defaultVersion?: string;
  supportedVersions: string[];
  deprecatedVersions?: Map<string, Date>;
}

interface VersionedRoute {
  version: string;
  handler: (req: Request, res: Response, next: NextFunction) => void | Promise<void>;
  deprecated?: boolean;
  sunsetDate?: Date;
}

export class ApiVersioningService {
  private config: VersionConfig;
  private routes: Map<string, VersionedRoute[]> = new Map();

  constructor(config: Partial<VersionConfig> = {}) {
    this.config = {
      strategy: config.strategy || VersionStrategy.HEADER,
      headerName: config.headerName || 'X-API-Version',
      queryParam: config.queryParam || 'version',
      defaultVersion: config.defaultVersion || 'v1',
      supportedVersions: config.supportedVersions || ['v1'],
      deprecatedVersions: config.deprecatedVersions || new Map()
    };
  }

  extractVersion(req: Request): string {
    let version: string | undefined;

    switch (this.config.strategy) {
      case VersionStrategy.HEADER:
        version = req.headers[this.config.headerName!.toLowerCase()] as string;
        break;

      case VersionStrategy.URL:
        const urlMatch = req.path.match(/^\/(v\d+)\//);
        version = urlMatch ? urlMatch[1] : undefined;
        break;

      case VersionStrategy.QUERY:
        version = req.query[this.config.queryParam!] as string;
        break;

      case VersionStrategy.ACCEPT_HEADER:
        const acceptHeader = req.headers.accept;
        if (acceptHeader) {
          const versionMatch = acceptHeader.match(/application\/vnd\.api\.(v\d+)\+json/);
          version = versionMatch ? versionMatch[1] : undefined;
        }
        break;
    }

    return version || this.config.defaultVersion!;
  }

  middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      const version = this.extractVersion(req);

      if (!this.isVersionSupported(version)) {
        return res.status(400).json({
          error: 'Unsupported API Version',
          message: `Version ${version} is not supported`,
          supportedVersions: this.config.supportedVersions
        });
      }

      (req as any).apiVersion = version;
      res.setHeader('X-API-Version', version);

      if (this.isVersionDeprecated(version)) {
        const sunsetDate = this.config.deprecatedVersions!.get(version);
        res.setHeader('Deprecation', 'true');
        if (sunsetDate) {
          res.setHeader('Sunset', sunsetDate.toUTCString());
        }
        res.setHeader('Link', `</docs/migration>; rel="deprecation"`);
      }

      next();
    };
  }

  registerVersionedRoute(
    path: string,
    version: string,
    handler: (req: Request, res: Response, next: NextFunction) => void | Promise<void>,
    options?: { deprecated?: boolean; sunsetDate?: Date }
  ): void {
    if (!this.routes.has(path)) {
      this.routes.set(path, []);
    }

    const routes = this.routes.get(path)!;
    routes.push({
      version,
      handler,
      deprecated: options?.deprecated,
      sunsetDate: options?.sunsetDate
    });

    routes.sort((a, b) => {
      const aNum = parseInt(a.version.replace('v', ''));
      const bNum = parseInt(b.version.replace('v', ''));
      return bNum - aNum;
    });
  }

  getVersionedHandler(path: string, version: string): VersionedRoute | undefined {
    const routes = this.routes.get(path);
    if (!routes) return undefined;

    return routes.find(route => route.version === version) || 
           routes.find(route => route.version === this.config.defaultVersion);
  }

  private isVersionSupported(version: string): boolean {
    return this.config.supportedVersions.includes(version);
  }

  private isVersionDeprecated(version: string): boolean {
    return this.config.deprecatedVersions!.has(version);
  }

  deprecateVersion(version: string, sunsetDate?: Date): void {
    this.config.deprecatedVersions!.set(version, sunsetDate || new Date(Date.now() + 90 * 24 * 60 * 60 * 1000));
  }

  removeVersion(version: string): void {
    const index = this.config.supportedVersions.indexOf(version);
    if (index > -1) {
      this.config.supportedVersions.splice(index, 1);
    }
    this.config.deprecatedVersions!.delete(version);
  }

  addVersion(version: string): void {
    if (!this.config.supportedVersions.includes(version)) {
      this.config.supportedVersions.push(version);
      this.config.supportedVersions.sort((a, b) => {
        const aNum = parseInt(a.replace('v', ''));
        const bNum = parseInt(b.replace('v', ''));
        return aNum - bNum;
      });
    }
  }

  getVersionInfo(): {
    current: string;
    supported: string[];
    deprecated: Array<{ version: string; sunsetDate?: Date }>;
  } {
    const deprecated = Array.from(this.config.deprecatedVersions!.entries()).map(
      ([version, sunsetDate]) => ({ version, sunsetDate })
    );

    return {
      current: this.config.supportedVersions[this.config.supportedVersions.length - 1],
      supported: this.config.supportedVersions,
      deprecated
    };
  }

  generateMigrationGuide(fromVersion: string, toVersion: string): string {
    const fromNum = parseInt(fromVersion.replace('v', ''));
    const toNum = parseInt(toVersion.replace('v', ''));

    if (fromNum >= toNum) {
      return 'No migration needed - you are on the latest or a newer version.';
    }

    const changes: string[] = [];
    
    for (let v = fromNum + 1; v <= toNum; v++) {
      changes.push(`\n### Migration from v${v-1} to v${v}:`);
      changes.push(this.getVersionChanges(`v${v}`));
    }

    return changes.join('\n');
  }

  private getVersionChanges(version: string): string {
    const changeLog: Record<string, string> = {
      v2: `
- Breaking: Authentication now requires JWT tokens instead of API keys
- New: Added support for biometric authentication endpoints
- Changed: Rate limiting now applies per user instead of per IP
- Deprecated: /auth/login endpoint (use /auth/token instead)
      `,
      v3: `
- Breaking: All timestamps now in ISO 8601 format
- New: GraphQL endpoint available at /graphql
- Changed: Pagination now uses cursor-based instead of offset
- Removed: Legacy /v1/users endpoint
      `,
      v4: `
- Breaking: Minimum TLS version is now 1.3
- New: WebSocket support for real-time fraud alerts
- Changed: Error responses now follow RFC 7807 Problem Details
- New: Batch operations endpoint for bulk processing
      `
    };

    return changeLog[version] || 'No documented changes for this version.';
  }
}

export const createVersioningService = (config?: Partial<VersionConfig>): ApiVersioningService => {
  return new ApiVersioningService(config);
};
