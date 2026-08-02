# 🎓 SmartClass — React App

HTML se React mein convert kiya gaya SmartClass Cheating Detection System.

## Project Structure

```
src/
├── api.js                   # Shared API utility functions (apiGet, apiPost)
├── App.js                   # Main app with React Router
├── index.js                 # Entry point
├── index.css                # Global CSS styles
├── components/
│   ├── Navbar.js            # Shared navigation bar
│   ├── LogContainer.js      # Activity log component
│   └── Toast.js             # Toast notification system
└── pages/
    ├── Dashboard.js         # /  — Main dashboard page
    └── Monitor.js           # /monitor — Live monitoring page
```

## Setup & Run

```bash
npm install
npm start
```

## Build for Production

```bash
npm run build
```

## API Endpoints (Backend se connect karo)

| Endpoint                  | Method | Description          |
|---------------------------|--------|----------------------|
| `/api/status`             | GET    | Model status         |
| `/api/stats`              | GET    | Live stats + logs    |
| `/api/monitoring/start`   | POST   | Start monitoring     |
| `/api/monitoring/stop`    | POST   | Stop monitoring      |
| `/api/video_feed`         | GET    | MJPEG video stream   |
| `/api/export_report`      | GET    | Download CSV report  |

## Notes

- Dashboard `/` — Model status, live stats, risk table, activity log
- Monitor `/monitor` — Live video feed, real-time alerts, student list
- Polling interval: 1500ms (same as original HTML)
- Toast notifications auto-dismiss after 4 seconds with debounce
