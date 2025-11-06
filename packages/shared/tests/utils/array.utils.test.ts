import { describe, it, expect } from '@jest/globals';
import { ArrayUtils } from '../../src/utils/array.utils';

describe('ArrayUtils', () => {
  describe('chunk', () => {
    it('should split array into chunks', () => {
      const array = [1, 2, 3, 4, 5, 6, 7];
      const result = ArrayUtils.chunk(array, 3);
      expect(result).toEqual([[1, 2, 3], [4, 5, 6], [7]]);
    });

    it('should handle empty array', () => {
      const result = ArrayUtils.chunk([], 3);
      expect(result).toEqual([]);
    });
  });

  describe('unique', () => {
    it('should remove duplicates', () => {
      const array = [1, 2, 2, 3, 3, 3, 4];
      const result = ArrayUtils.unique(array);
      expect(result).toEqual([1, 2, 3, 4]);
    });
  });

  describe('groupBy', () => {
    it('should group array by key', () => {
      const array = [
        { category: 'A', value: 1 },
        { category: 'B', value: 2 },
        { category: 'A', value: 3 },
      ];
      const result = ArrayUtils.groupBy(array, 'category');
      expect(result).toEqual({
        A: [
          { category: 'A', value: 1 },
          { category: 'A', value: 3 },
        ],
        B: [{ category: 'B', value: 2 }],
      });
    });
  });

  describe('shuffle', () => {
    it('should shuffle array', () => {
      const array = [1, 2, 3, 4, 5];
      const result = ArrayUtils.shuffle(array);
      expect(result).toHaveLength(5);
      expect(result.sort()).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('intersection', () => {
    it('should find common elements', () => {
      const array1 = [1, 2, 3, 4];
      const array2 = [3, 4, 5, 6];
      const result = ArrayUtils.intersection(array1, array2);
      expect(result).toEqual([3, 4]);
    });
  });

  describe('difference', () => {
    it('should find elements in first array not in second', () => {
      const array1 = [1, 2, 3, 4];
      const array2 = [3, 4, 5, 6];
      const result = ArrayUtils.difference(array1, array2);
      expect(result).toEqual([1, 2]);
    });
  });

  describe('partition', () => {
    it('should partition array by predicate', () => {
      const array = [1, 2, 3, 4, 5, 6];
      const [even, odd] = ArrayUtils.partition(array, (n) => n % 2 === 0);
      expect(even).toEqual([2, 4, 6]);
      expect(odd).toEqual([1, 3, 5]);
    });
  });

  describe('statistics', () => {
    const numbers = [1, 2, 3, 4, 5];

    it('should calculate sum', () => {
      expect(ArrayUtils.sum(numbers)).toBe(15);
    });

    it('should calculate average', () => {
      expect(ArrayUtils.average(numbers)).toBe(3);
    });

    it('should calculate median', () => {
      expect(ArrayUtils.median(numbers)).toBe(3);
      expect(ArrayUtils.median([1, 2, 3, 4])).toBe(2.5);
    });

    it('should find min and max', () => {
      expect(ArrayUtils.min(numbers)).toBe(1);
      expect(ArrayUtils.max(numbers)).toBe(5);
    });
  });

  describe('range', () => {
    it('should generate range of numbers', () => {
      const result = ArrayUtils.range(0, 5);
      expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it('should generate range with step', () => {
      const result = ArrayUtils.range(0, 10, 2);
      expect(result).toEqual([0, 2, 4, 6, 8]);
    });
  });
});
