export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResult<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export function createPaginatedResult<T>(
  data: T[],
  total: number,
  params: PaginationParams
): PaginatedResult<T> {
  const totalPages = Math.ceil(total / params.limit);

  return {
    data,
    pagination: {
      page: params.page,
      limit: params.limit,
      total,
      totalPages,
      hasNext: params.page < totalPages,
      hasPrev: params.page > 1
    }
  };
}

export function calculateOffset(page: number, limit: number): number {
  return (page - 1) * limit;
}

export function validatePaginationParams(params: Partial<PaginationParams>): PaginationParams {
  const page = Math.max(1, params.page || 1);
  const limit = Math.min(100, Math.max(1, params.limit || 20));

  return {
    page,
    limit,
    sortBy: params.sortBy,
    sortOrder: params.sortOrder || 'desc'
  };
}

export interface CursorPaginationParams {
  cursor?: string;
  limit: number;
}

export interface CursorPaginatedResult<T> {
  data: T[];
  nextCursor?: string;
  hasMore: boolean;
}

export function encodeCursor(value: string | number): string {
  return Buffer.from(String(value)).toString('base64url');
}

export function decodeCursor(cursor: string): string {
  return Buffer.from(cursor, 'base64url').toString('utf-8');
}
