import React, { useState, useEffect } from 'react';

interface Transaction {
  id: string;
  userId: string;
  amount: number;
  currency: string;
  merchantId: string;
  riskScore: number;
  status: string;
  timestamp: Date;
}

export const TransactionList: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filter, setFilter] = useState<'all' | 'flagged' | 'approved' | 'declined'>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTransactions();
  }, [filter]);

  const fetchTransactions = async () => {
    try {
      const response = await fetch(`/api/transactions?filter=${filter}&limit=50`);
      const data = await response.json();
      setTransactions(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch transactions:', error);
      setLoading(false);
    }
  };

  const getRiskColor = (riskScore: number): string => {
    if (riskScore >= 0.7) return 'high-risk';
    if (riskScore >= 0.4) return 'medium-risk';
    return 'low-risk';
  };

  const getStatusBadge = (status: string): string => {
    const statusMap: Record<string, string> = {
      approved: 'status-approved',
      declined: 'status-declined',
      flagged: 'status-flagged',
      pending: 'status-pending'
    };
    return statusMap[status] || 'status-default';
  };

  if (loading) {
    return <div className="loading">Loading transactions...</div>;
  }

  return (
    <div className="transaction-list">
      <div className="transaction-list-header">
        <h2>Recent Transactions</h2>
        <div className="filter-buttons">
          <button onClick={() => setFilter('all')} className={filter === 'all' ? 'active' : ''}>
            All
          </button>
          <button onClick={() => setFilter('flagged')} className={filter === 'flagged' ? 'active' : ''}>
            Flagged
          </button>
          <button onClick={() => setFilter('approved')} className={filter === 'approved' ? 'active' : ''}>
            Approved
          </button>
          <button onClick={() => setFilter('declined')} className={filter === 'declined' ? 'active' : ''}>
            Declined
          </button>
        </div>
      </div>

      <table className="transaction-table">
        <thead>
          <tr>
            <th>Transaction ID</th>
            <th>User ID</th>
            <th>Amount</th>
            <th>Merchant</th>
            <th>Risk Score</th>
            <th>Status</th>
            <th>Timestamp</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction) => (
            <tr key={transaction.id}>
              <td>{transaction.id}</td>
              <td>{transaction.userId}</td>
              <td>
                {transaction.amount.toFixed(2)} {transaction.currency}
              </td>
              <td>{transaction.merchantId}</td>
              <td>
                <span className={`risk-badge ${getRiskColor(transaction.riskScore)}`}>
                  {(transaction.riskScore * 100).toFixed(0)}%
                </span>
              </td>
              <td>
                <span className={`status-badge ${getStatusBadge(transaction.status)}`}>
                  {transaction.status}
                </span>
              </td>
              <td>{new Date(transaction.timestamp).toLocaleString()}</td>
              <td>
                <button className="btn-view">View</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
