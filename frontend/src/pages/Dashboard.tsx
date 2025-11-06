import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';  // Assume Chart.js integration
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { biometricService } from '../services/biometricService';
import type { FraudResponse } from '../types';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

interface DashboardProps {
  alerts: FraudResponse[];
}

const Dashboard: React.FC<DashboardProps> = ({ alerts }) => {
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalVerifications: 0,
    fraudRate: 0,
    avgScore: 0
  });
  const [recentActivity, setRecentActivity] = useState<string[]>([]);

  useEffect(() => {
    // Simulate stats calculation
    const frauds = alerts.filter(a => a.fraud_detected).length;
    const total = alerts.length || 1;
    setStats({
      totalUsers: 150,  // Mock
      totalVerifications: total,
      fraudRate: (frauds / total * 100).toFixed(2),
      avgScore: alerts.reduce((sum, a) => sum + (a.score || 0), 0) / total
    });

    // Recent activity log
    setRecentActivity([
      'User_123 verified successfully',
      'Fraud alert for User_456 - high score',
      'Biometric enrollment for new user',
      // ... more logs for volume
      ...Array(20).fill('System monitoring active...').map((msg, i) => `${msg} [${i}]`)
    ]);
  }, [alerts]);

  // Chart data for fraud trends
  const fraudChartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Fraud Incidents',
        data: [12, 19, 3, 5, 2, 3],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
      },
      {
        label: 'Verifications',
        data: [200, 250, 220, 300, 280, 310],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
      }
    ]
  };

  const fraudChartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' as const },
      title: { display: true, text: 'Fraud Trends Over Time' }
    }
  };

  // User metrics bar chart
  const metricsData = {
    labels: ['Enrolled Users', 'Pending', 'Fraud Detected', 'Verified'],
    datasets: [{
      label: 'Count',
      data: [120, 30, 15, 1000],
      backgroundColor: ['rgba(54, 162, 235, 0.8)', 'rgba(255, 206, 86, 0.8)', 'rgba(255, 99, 132, 0.8)', 'rgba(75, 192, 192, 0.8)']
    }]
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Biometric Fraud Prevention Dashboard</h1>
        <p>Real-time monitoring and analytics | Last updated: {new Date().toLocaleString()}</p>
      </header>

      <section className="stats-grid">
        <div className="stat-card">
          <h3>Total Users</h3>
          <p className="stat-value">{stats.totalUsers}</p>
        </div>
        <div className="stat-card">
          <h3>Fraud Rate</h3>
          <p className="stat-value danger">{stats.fraudRate}%</p>
        </div>
        <div className="stat-card">
          <h3>Avg Fraud Score</h3>
          <p className="stat-value">{stats.avgScore.toFixed(3)}</p>
        </div>
        <div className="stat-card">
          <h3>Verifications Today</h3>
          <p className="stat-value">{stats.totalVerifications}</p>
        </div>
      </section>

      <section className="charts-section">
        <div className="chart-container">
          <Line data={fraudChartData} options={fraudChartOptions} />
        </div>
        <div className="chart-container">
          <Bar data={metricsData} options={{ responsive: true, plugins: { title: { display: true, text: 'User Metrics' } } }} />
        </div>
      </section>

      <section className="activity-log">
        <h2>Recent Activity</h2>
        <ul className="activity-list">
          {recentActivity.slice(0, 10).map((activity, index) => (
            <li key={index} className="activity-item">{activity}</li>
          ))}
        </ul>
      </section>

      <footer className="dashboard-footer">
        <p>System Status: Operational | Alerts: {alerts.length}</p>
      </footer>
    </div>
  );
};

export default Dashboard;
