import { describe, it, expect } from '@jest/globals';
import {
  createPaginatedResult,
  calculateOffset,
  validatePaginationParams,
  encodeCursor,
  decodeCursor
} from '../../src/utils/pagination.utils';

describe('Pagination Utils', () => {
  describe('createPaginatedResult', () => {
    it('should create paginated result with correct metadata', () => {
      const data = [1, 2, 3, 4, 5];
      const total = 50;
      const params = { page: 1, limit: 5 };

      const result = createPaginatedResult(data, total, params);

      expect(result.data).toEqual(data);
      expect(result.pagination.page).toBe(1);
      expect(result.pagination.limit).toBe(5);
      expect(result.pagination.total).toBe(50);
      expect(result.pagination.totalPages).toBe(10);
      expect(result.pagination.hasNext).toBe(true);
      expect(result.pagination.hasPrev).toBe(false);
    });

    it('should indicate no next page on last page', () => {
      const data = [1, 2, 3];
      const total = 23;
      const params = { page: 5, limit: 5 };

      const result = createPaginatedResult(data, total, params);

      expect(result.pagination.hasNext).toBe(false);
      expect(result.pagination.hasPrev).toBe(true);
    });
  });

  describe('calculateOffset', () => {
    it('should calculate correct offset', () => {
      expect(calculateOffset(1, 10)).toBe(0);
      expect(calculateOffset(2, 10)).toBe(10);
      expect(calculateOffset(3, 20)).toBe(40);
    });
  });

  describe('validatePaginationParams', () => {
    it('should validate and normalize pagination params', () => {
      const params = validatePaginationParams({ page: 2, limit: 50 });

      expect(params.page).toBe(2);
      expect(params.limit).toBe(50);
      expect(params.sortOrder).toBe('desc');
    });

    it('should enforce minimum page number', () => {
      const params = validatePaginationParams({ page: -1, limit: 10 });

      expect(params.page).toBe(1);
    });

    it('should enforce maximum limit', () => {
      const params = validatePaginationParams({ page: 1, limit: 200 });

      expect(params.limit).toBe(100);
    });

    it('should use default values', () => {
      const params = validatePaginationParams({});

      expect(params.page).toBe(1);
      expect(params.limit).toBe(20);
    });
  });

  describe('cursor encoding/decoding', () => {
    it('should encode and decode cursor', () => {
      const original = 'user-123-timestamp-456';
      const encoded = encodeCursor(original);
      const decoded = decodeCursor(encoded);

      expect(decoded).toBe(original);
    });

    it('should handle numeric cursors', () => {
      const original = 12345;
      const encoded = encodeCursor(original);
      const decoded = decodeCursor(encoded);

      expect(decoded).toBe('12345');
    });
  });
});
