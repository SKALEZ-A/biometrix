import React, { useEffect, useState } from 'react';

interface Alert {
  id: string;
  type: string;
  severity: string;
  title: string;
  description: string;
  status: string;
  createdAt: string;
}

export const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 15000);
    return () => clearInterval(interval);
  }, [filter]);

  const fetchAlerts = async () => {
    try {
      const url = filter === 'all' 
        ? '/api/alerts?status=open' 
        : `/api/alerts?status=open&severity=${filter}`;
      const response = await fetch(url);
      const data = await response.json();
      setAlerts(data.alerts);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      await fetch(`/api/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acknowledgedBy: 'current-user' })
      });
      fetchAlerts();
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  return (
    <div className="alerts-panel">
      <div className="alerts-header">
        <h2>Active Alerts</h2>
        <div className="filter-buttons">
          <button onClick={() => setFilter('all')} className={filter === 'all' ? 'active' : ''}>
            All
          </button>
          <button onClick={() => setFilter('critical')} className={filter === 'critical' ? 'active' : ''}>
            Critical
          </button>
          <button onClick={() => setFilter('high')} className={filter === 'high' ? 'active' : ''}>
            High
          </button>
        </div>
      </div>
      <div className="alerts-list">
        {alerts.map(alert => (
          <div key={alert.id} className={`alert-card severity-${alert.severity}`}>
            <div className="alert-header">
              <span className="alert-type">{alert.type}</span>
              <span className="alert-severity">{alert.severity}</span>
            </div>
            <h3>{alert.title}</h3>
            <p>{alert.description}</p>
            <div className="alert-footer">
              <span className="alert-time">
                {new Date(alert.createdAt).toLocaleString()}
              </span>
              <button onClick={() => handleAcknowledge(alert.id)}>
                Acknowledge
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
