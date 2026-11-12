"""
One-time helper: copy the accounts and branding you already have in users.json,
admins.json and branding.json into the database named by DATABASE_URL.

    export DATABASE_URL="postgresql://user:password@host/dbname"
    python3 migrate_to_db.py

It never overwrites anything that is already in the database, so it is safe to run twice.
(PDF uploads are not migrated: upload them again from the Admin page once the database is connected.)
"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()
if not os.getenv("DATABASE_URL"):
    sys.exit("Set DATABASE_URL first (in .env or the terminal), then run this again.")

from storage import storage, BASE_DIR  # noqa: E402


def read(name):
    try:
        with open(os.path.join(BASE_DIR, name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


users, admins, branding = read("users.json"), read("admins.json"), read("branding.json")

added_users = sum(1 for u, rec in users.items() if storage.create_user(u, rec))
added_admins = sum(1 for u, rec in admins.items() if storage.create_admin(u, rec))
print(f"Database: {storage.name}")
print(f"Users copied:  {added_users} of {len(users)} (others already existed)")
print(f"Admins copied: {added_admins} of {len(admins)} (others already existed)")

if branding.get("name"):
    current = storage.get_branding()
    if current["name"] == "NEXUS" and not current["tagline"]:
        storage.set_branding(branding["name"], branding.get("tagline", ""))
        print(f"Branding copied: {branding['name']!r}")
    else:
        print("Branding already set in the database; left unchanged.")
