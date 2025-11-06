import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { Dashboard } from '../src/components/Dashboard';

global.fetch = jest.fn();

describe('Dashboard Component', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear();
  });

  it('should render loading state initially', () => {
    (fetch as jest.Mock).mockImplementation(() =>
      new Promise(() => {}) // Never resolves
    );

    render(<Dashboard />);

    expect(screen.getByText('Loading dashboard...')).toBeInTheDocument();
  });

  it('should render dashboard stats after loading', async () => {
    const mockStats = {
      totalTransactions: 1000,
      fraudDetected: 25,
      averageRiskScore: 35.5,
      activeAlerts: 5
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStats
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('1,000')).toBeInTheDocument();
      expect(screen.getByText('25')).toBeInTheDocument();
      expect(screen.getByText('35.50')).toBeInTheDocument();
      expect(screen.getByText('5')).toBeInTheDocument();
    });
  });

  it('should display error message on fetch failure', async () => {
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    render(<Dashboard />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to fetch dashboard stats:',
        expect.any(Error)
      );
    });

    consoleSpy.mockRestore();
  });
});
