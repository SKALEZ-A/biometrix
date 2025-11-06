export class GeoUtils {
  private static readonly EARTH_RADIUS_KM = 6371;

  static calculateDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const dLat = this.toRadians(lat2 - lat1);
    const dLon = this.toRadians(lon2 - lon1);

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRadians(lat1)) *
        Math.cos(this.toRadians(lat2)) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return this.EARTH_RADIUS_KM * c;
  }

  static toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  static toDegrees(radians: number): number {
    return radians * (180 / Math.PI);
  }

  static isWithinRadius(
    centerLat: number,
    centerLon: number,
    pointLat: number,
    pointLon: number,
    radiusKm: number
  ): boolean {
    const distance = this.calculateDistance(centerLat, centerLon, pointLat, pointLon);
    return distance <= radiusKm;
  }

  static getBoundingBox(
    lat: number,
    lon: number,
    radiusKm: number
  ): { minLat: number; maxLat: number; minLon: number; maxLon: number } {
    const latDelta = radiusKm / this.EARTH_RADIUS_KM;
    const lonDelta = radiusKm / (this.EARTH_RADIUS_KM * Math.cos(this.toRadians(lat)));

    return {
      minLat: lat - this.toDegrees(latDelta),
      maxLat: lat + this.toDegrees(latDelta),
      minLon: lon - this.toDegrees(lonDelta),
      maxLon: lon + this.toDegrees(lonDelta)
    };
  }
}
