"""
Database Schema Manager for the Text-to-SQL system.

Manages database schema information for a given SQLite database.
"""
import sqlite3

def describe_database(db_path: str) -> str:
    """Return a human-readable schema description for a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    lines: list[str] = []
    for table in tables:
        lines.append(f"\nTable: {table}")

        cursor.execute(f"PRAGMA table_info({table});")
        for row in cursor.fetchall():
            col_name, col_type = row[1], row[2]
            lines.append(f"  - {col_name} ({col_type})")

        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        if fks:
            lines.append("  Foreign Keys:")
            for fk in fks:
                from_col, ref_table, ref_col = fk[3], fk[2], fk[4]
                lines.append(f"    {from_col} → {ref_table}.{ref_col}")

    conn.close()
    return "\n".join(lines)
