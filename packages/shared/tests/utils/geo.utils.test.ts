import { GeoUtils } from '../../src/utils/geo.utils';

describe('GeoUtils', () => {
  describe('calculateDistance', () => {
    it('should calculate distance between two points', () => {
      const distance = GeoUtils.calculateDistance(40.7128, -74.0060, 34.0522, -118.2437);
      expect(distance).toBeGreaterThan(3900);
      expect(distance).toBeLessThan(4000);
    });

    it('should return 0 for same coordinates', () => {
      const distance = GeoUtils.calculateDistance(40.7128, -74.0060, 40.7128, -74.0060);
      expect(distance).toBe(0);
    });
  });

  describe('isWithinRadius', () => {
    it('should return true for points within radius', () => {
      const result = GeoUtils.isWithinRadius(40.7128, -74.0060, 40.7580, -73.9855, 10);
      expect(result).toBe(true);
    });

    it('should return false for points outside radius', () => {
      const result = GeoUtils.isWithinRadius(40.7128, -74.0060, 34.0522, -118.2437, 100);
      expect(result).toBe(false);
    });
  });

  describe('getBoundingBox', () => {
    it('should calculate bounding box', () => {
      const box = GeoUtils.getBoundingBox(40.7128, -74.0060, 10);
      
      expect(box).toHaveProperty('minLat');
      expect(box).toHaveProperty('maxLat');
      expect(box).toHaveProperty('minLon');
      expect(box).toHaveProperty('maxLon');
      expect(box.minLat).toBeLessThan(40.7128);
      expect(box.maxLat).toBeGreaterThan(40.7128);
    });
  });

  describe('toRadians', () => {
    it('should convert degrees to radians', () => {
      expect(GeoUtils.toRadians(180)).toBeCloseTo(Math.PI);
      expect(GeoUtils.toRadians(90)).toBeCloseTo(Math.PI / 2);
      expect(GeoUtils.toRadians(0)).toBe(0);
    });
  });

  describe('toDegrees', () => {
    it('should convert radians to degrees', () => {
      expect(GeoUtils.toDegrees(Math.PI)).toBeCloseTo(180);
      expect(GeoUtils.toDegrees(Math.PI / 2)).toBeCloseTo(90);
      expect(GeoUtils.toDegrees(0)).toBe(0);
    });
  });
});
