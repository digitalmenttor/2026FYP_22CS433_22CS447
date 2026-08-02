import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import { ThemeProvider } from './ThemeContext.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Monitor from './pages/Monitor.jsx';
import Register from './pages/Register.jsx';
import Students from './pages/Students.jsx';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/register" element={<Register />} />
          <Route path="/students" element={<Students />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
