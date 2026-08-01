# Python shell mein:
import database as db
db.init_db()
sid = db.start_session()
print("Session ID:", sid)
db.log_cheating_event(sid, 1, "Test Student", "yaw", severity="HIGH")
print("Event logged!")
db.end_session(sid, {'session_time': 10, 'person_count': 1,
                      'total_yaw_alerts': 1, 'total_peeking_alerts': 0,
                      'total_mobile_detections': 0, 'total_paper_exchanges': 0})
print("Done!")