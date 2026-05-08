#!/usr/bin/env python3
"""
gen_12_inter_group.py
=====================
Inter-group multi-tool scenarios using the REAL Suwon catalog tools
(from tools_summary.md). Every seed forces tool calls that span
≥2 distinct groups.

Group naming below uses the markdown section headers, lowercased and
underscored. Make sure these names match the GROUP_NAME values in your
gen_01..gen_11 scripts — if any of them differ, just rename the keys
in ALLOWLISTS_BY_GROUP to match.

Themes covered:
    A. battery_power     + communication      ("phone dying, text my friend")
    B. system_device     + communication      ("share my diagnostics with IT")
    C. personal_info     + display_ui         ("add exam tomorrow, dim screen")
    D. connectivity      + battery_power      ("WiFi off, log battery")
    E. notifications     + personal_info      ("mute except alarms for 1h")
    F. clipboard         + communication      ("copy this and share it")
    G. memory_files      + communication      ("find the report and share")
    H. apps              + personal_info      ("set alarm, open Strava")
    I. audio             + notifications      ("silent mode, check DND")
    J. battery_power     + display_ui         ("dim screen, save battery")
    K. THREE-GROUP scenario                   ("exam mode": personal+audio+notif)
"""
from generator_core import run_intergroup


GROUP_NAME = "inter_group"

# Maps group_name -> set of tool names from that group eligible for cross-group
# use. Names taken DIRECTLY from tools_summary.md.
ALLOWLISTS_BY_GROUP = {
    "system_device": {
        "system_settings_get", "system_settings_set", "device_info",
        "system", "diagnostics", "sensors",
    },
    "display_ui": {
        "display_control", "display_status", "font_size_get", "font_size_set",
    },
    "audio": {
        "audio_control", "audio_status",
    },
    "apps": {
        "apps_info", "apps_manage", "apps_launch",
    },
    "connectivity": {
        "connectivity_control", "connectivity_status",
        "location_control", "location_status",
    },
    "battery_power": {
        "battery", "device_control",
    },
    "memory_files": {
        "memory_read", "memory_search", "memory_write", "files", "storage",
    },
    "personal_info": {
        "calendar", "alarms", "accounts",
    },
    "communication": {
        "phone", "share", "intent",
    },
    "clipboard": {
        "clipboard_get", "clipboard_set",
    },
    "notifications": {
        "notifications_status", "vibration",
    },
}


SEEDS = [
    # ------ A. battery_power + communication ----------------------------
    {
        "intent": "My phone's about to die — let my brother know to call me later.",
        "target_tools": ["battery", "phone"],
        "groups_required": ["battery_power", "communication"],
        "hint": "Read battery first to confirm low, then dial the brother contact.",
        "persona": "casual",
        "expected_outcome": "Battery checked and call placed/queued to brother.",
    },
    {
        "intent": "Battery is critical — share my battery status with my partner.",
        "target_tools": ["battery", "share"],
        "groups_required": ["battery_power", "communication"],
        "hint": "Read battery, then use share to send level via the chooser.",
        "persona": "worried",
        "expected_outcome": "Battery level read and a share intent prepared.",
    },

    # ------ B. system_device + communication ----------------------------
    {
        "intent": "IT wants my device diagnostics — send them over.",
        "target_tools": ["diagnostics", "device_info", "share"],
        "groups_required": ["system_device", "communication"],
        "hint": "Pull diagnostics + device_info, then share the combined info.",
        "persona": "professional",
        "expected_outcome": "Diagnostics + device info collected and shared.",
    },

    # ------ C. personal_info + display_ui -------------------------------
    {
        "intent": "I have an exam tomorrow morning — set it up and make my screen easier on the eyes.",
        "target_tools": ["calendar", "alarms", "display_control"],
        "groups_required": ["personal_info", "display_ui"],
        "hint": "Calendar entry for the exam, alarm before it, dim brightness now.",
        "persona": "anxious_student",
        "expected_outcome": "Calendar event + alarm created and brightness lowered.",
    },
    {
        "intent": "Add band practice every Wednesday and bump the font size — I keep missing it.",
        "target_tools": ["calendar", "font_size_set"],
        "groups_required": ["personal_info", "display_ui"],
        "hint": "Recurring calendar entry plus a larger font scale.",
        "persona": "older_user",
        "expected_outcome": "Recurring event created and font size increased.",
    },

    # ------ D. connectivity + battery_power -----------------------------
    {
        "intent": "Turn off WiFi to save battery and log my current battery level.",
        "target_tools": ["connectivity_control", "battery"],
        "groups_required": ["connectivity", "battery_power"],
        "hint": "Toggle wifi off via connectivity_control, then read battery.",
        "persona": "data-conscious",
        "expected_outcome": "WiFi disabled and battery level reported.",
    },
    {
        "intent": "Are location services on? If yes, turn them off and check device storage.",
        "target_tools": ["location_status", "location_control", "storage"],
        "groups_required": ["connectivity", "memory_files"],
        "hint": "Conditional: status check, then maybe-disable location, then storage.",
        "persona": "privacy-minded",
        "expected_outcome": "Location state confirmed/disabled and storage reported.",
    },

    # ------ E. notifications + personal_info ----------------------------
    {
        "intent": "I'm in a deep focus session — silence everything for an hour but keep alarms.",
        "target_tools": ["notifications_status", "system_settings_set", "alarms"],
        "groups_required": ["notifications", "personal_info"],
        "hint": "Check DND, enable it via system_settings_set, verify alarm is set.",
        "persona": "focused_professional",
        "expected_outcome": "DND on, existing alarm preserved.",
    },

    # ------ F. clipboard + communication --------------------------------
    {
        "intent": "Copy this address and send it to the team.",
        "target_tools": ["clipboard_set", "share"],
        "groups_required": ["clipboard", "communication"],
        "hint": "Set clipboard with the address, then trigger share.",
        "persona": "casual",
        "expected_outcome": "Clipboard updated and share sheet opened.",
    },
    {
        "intent": "What's in my clipboard? If it's a phone number, dial it.",
        "target_tools": ["clipboard_get", "phone"],
        "groups_required": ["clipboard", "communication"],
        "hint": "Read clipboard, conditionally call phone.",
        "persona": "casual",
        "expected_outcome": "Clipboard checked; call placed if it was a number.",
    },

    # ------ G. memory_files + communication -----------------------------
    {
        "intent": "Find my Q3 report in Downloads and share it.",
        "target_tools": ["memory_search", "files", "share"],
        "groups_required": ["memory_files", "communication"],
        "hint": "Search memory, list Downloads, share the matched file.",
        "persona": "professional",
        "expected_outcome": "Q3 report located and shared.",
    },

    # ------ H. apps + personal_info -------------------------------------
    {
        "intent": "Set a 30-minute alarm and open my running app.",
        "target_tools": ["alarms", "apps_info", "apps_launch"],
        "groups_required": ["personal_info", "apps"],
        "hint": "Alarm 30 min out, look up running app via apps_info, launch it.",
        "persona": "active",
        "expected_outcome": "Alarm scheduled and running app launched.",
    },
    {
        "intent": "Open Calendar and add a dentist appointment.",
        "target_tools": ["apps_launch", "calendar"],
        "groups_required": ["apps", "personal_info"],
        "hint": "Launch the Calendar app, then create the calendar event.",
        "persona": "neutral",
        "expected_outcome": "Calendar app opened and event created.",
    },

    # ------ I. audio + notifications ------------------------------------
    {
        "intent": "Going into a meeting — silent mode and confirm DND status.",
        "target_tools": ["audio_control", "notifications_status"],
        "groups_required": ["audio", "notifications"],
        "hint": "Switch ringer to silent, then read notifications_status to verify DND.",
        "persona": "professional",
        "expected_outcome": "Phone silenced and DND state confirmed.",
    },

    # ------ J. battery_power + display_ui -------------------------------
    {
        "intent": "Battery's at 12% — do whatever you can on the screen side to stretch it.",
        "target_tools": ["battery", "display_control", "display_status"],
        "groups_required": ["battery_power", "display_ui"],
        "hint": "Read battery, read display_status, then dim brightness + reduce timeout.",
        "persona": "frustrated",
        "expected_outcome": "Battery confirmed low, display dimmed, timeout reduced.",
    },

    # ------ K. THREE-GROUP scenario -------------------------------------
    {
        "intent": "I'm taking a 2-hour exam — set a calendar block, vibrate-only mode, and an alarm 5 min before it ends.",
        "target_tools": ["calendar", "audio_control", "vibration", "alarms"],
        "groups_required": ["personal_info", "audio", "notifications"],
        "hint": "Calendar block, ringer to vibrate, vibration test, alarm at end-5.",
        "persona": "student",
        "expected_outcome": "Calendar block + vibrate mode + alarm all set.",
    },
]


if __name__ == "__main__":
    run_intergroup(
        GROUP_NAME,
        ALLOWLISTS_BY_GROUP,
        SEEDS,
        "group_12_inter_group.jsonl",
    )
