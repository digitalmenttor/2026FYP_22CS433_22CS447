import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../ThemeContext.jsx';

export default function Navbar({ isLive }) {
  const { pathname } = useLocation();
  const { theme, toggle } = useTheme();

  const links = [
    { to: '/',          label: '📊 Dashboard' },
    { to: '/monitor',   label: '🎥 Monitor' },
    { to: '/register',  label: '📷 Register' },
    { to: '/students',  label: '👥 Students' },
  ];

  return (
    <nav className="nav">
      <div className="nav-logo">🎓 SmartClass</div>
      <div className="nav-links">
        {links.map(({ to, label }) => (
          <Link key={to} className={`nav-btn${pathname === to ? ' active' : ''}`} to={to}>
            {label}
          </Link>
        ))}
      </div>
      <button
        className="theme-toggle"
        onClick={toggle}
        title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        aria-label="Toggle theme"
      >
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
      <span className={`status-dot${isLive ? ' live' : ''}`} />
    </nav>
  );
}
