import { TimeUtils } from '../../src/utils/time.utils';

describe('TimeUtils', () => {
  describe('addDays', () => {
    it('should add days to date', () => {
      const date = new Date('2024-01-01');
      const result = TimeUtils.addDays(date, 5);
      expect(result.getDate()).toBe(6);
    });

    it('should handle negative days', () => {
      const date = new Date('2024-01-10');
      const result = TimeUtils.addDays(date, -5);
      expect(result.getDate()).toBe(5);
    });
  });

  describe('getDaysDifference', () => {
    it('should calculate days difference', () => {
      const date1 = new Date('2024-01-01');
      const date2 = new Date('2024-01-10');
      const diff = TimeUtils.getDaysDifference(date1, date2);
      expect(diff).toBe(9);
    });

    it('should return positive value regardless of order', () => {
      const date1 = new Date('2024-01-10');
      const date2 = new Date('2024-01-01');
      const diff = TimeUtils.getDaysDifference(date1, date2);
      expect(diff).toBe(9);
    });
  });

  describe('isWeekend', () => {
    it('should return true for Saturday', () => {
      const saturday = new Date('2024-01-06');
      expect(TimeUtils.isWeekend(saturday)).toBe(true);
    });

    it('should return true for Sunday', () => {
      const sunday = new Date('2024-01-07');
      expect(TimeUtils.isWeekend(sunday)).toBe(true);
    });

    it('should return false for weekday', () => {
      const monday = new Date('2024-01-08');
      expect(TimeUtils.isWeekend(monday)).toBe(false);
    });
  });

  describe('isBusinessHours', () => {
    it('should return true for business hours', () => {
      const date = new Date('2024-01-08 10:00:00');
      expect(TimeUtils.isBusinessHours(date)).toBe(true);
    });

    it('should return false for after hours', () => {
      const date = new Date('2024-01-08 20:00:00');
      expect(TimeUtils.isBusinessHours(date)).toBe(false);
    });

    it('should return false for weekend', () => {
      const date = new Date('2024-01-06 10:00:00');
      expect(TimeUtils.isBusinessHours(date)).toBe(false);
    });
  });

  describe('formatDuration', () => {
    it('should format seconds', () => {
      expect(TimeUtils.formatDuration(5000)).toBe('5s');
    });

    it('should format minutes', () => {
      expect(TimeUtils.formatDuration(125000)).toBe('2m 5s');
    });

    it('should format hours', () => {
      expect(TimeUtils.formatDuration(7325000)).toBe('2h 2m');
    });

    it('should format days', () => {
      expect(TimeUtils.formatDuration(90000000)).toBe('1d 1h');
    });
  });
});
