"""MoTeC .ldx XML beacon/lap export."""

from __future__ import annotations


def write_ldx(motec_log, filename, laps_info=None, beacons=None):
    """ Writes an XML .ldx file containing lap markers, sector beacons, and metadata details. """
    import xml.etree.ElementTree as ET

    root = ET.Element("LDXFile", {
        "Locale": "English_United States.1252",
        "DefaultLocale": "C",
        "Version": "1.6"
    })
    layers = ET.SubElement(root, "Layers")
    layer = ET.SubElement(layers, "Layer")

    marker_block = ET.SubElement(layer, "MarkerBlock")
    beacon_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Beacons", "Index": "3"})

    detected_lap_beacons = []
    split_beacons = []

    if beacons:
        for item in beacons:
            b_time = item[0] if isinstance(item, (tuple, list)) else item
            b_name = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else "Beacon"
            if str(b_name).strip().lower() in ("start/finish", "start", "finish", "sf", "beacon") or str(b_name).strip().lower().startswith("lap"):
                detected_lap_beacons.append((b_time, b_name))
            else:
                split_beacons.append((b_time, b_name))

    # Prefer source lap timing over GPS crossing detection. RCZ session.json
    # records millisecond-accurate boundaries, while detected beacons are
    # constrained to the resampled channel grid (typically 25 Hz).
    lap_beacons = []
    b_from_laps = laps_info.get("beacons", []) if laps_info else []
    if b_from_laps:
        lap_beacons = list(b_from_laps)
    else:
        laps = laps_info.get("laps", []) if laps_info else []
        timed_laps = [
            lap
            for lap in laps
            if str(lap.get("type", "Timed")).strip().lower() == "timed"
            and lap.get("start_time") is not None
            and lap.get("end_time") is not None
            and float(lap["end_time"]) > float(lap["start_time"])
        ]
        if timed_laps:
            boundaries = [float(timed_laps[0]["start_time"])]
            boundaries.extend(float(lap["end_time"]) for lap in timed_laps)
            for boundary in boundaries:
                if not lap_beacons or abs(boundary - lap_beacons[-1][0]) > 1e-9:
                    lap_beacons.append((boundary, "Start/Finish"))
        else:
            lap_beacons = detected_lap_beacons

    for idx, (b_time, b_name) in enumerate(lap_beacons):
        t_us = b_time * 1e6
        marker_name = f"Manual.{idx + 1}" if "manual" in str(b_name).lower() else str(b_name)
        ET.SubElement(beacon_group, "Marker", {
            "Version": "100",
            "ClassName": "BCN",
            "Name": marker_name,
            "Flags": "77",
            "Time": f"{t_us:.17e}"
        })

    if split_beacons:
        section_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Sections", "Index": "4"})
        for idx, (b_time, b_name) in enumerate(split_beacons):
            t_us = b_time * 1e6
            ET.SubElement(section_group, "Marker", {
                "Version": "100",
                "ClassName": "BCN",
                "Name": str(b_name),
                "Flags": "77",
                "Time": f"{t_us:.17e}"
            })

    ET.SubElement(layer, "RangeBlock")

    details = ET.SubElement(layers, "Details")

    def add_detail_str(string_id, val):
        ET.SubElement(details, "String", {"Id": string_id, "Value": str(val) if val is not None else ""})

    add_detail_str("Event", motec_log.event_name)
    add_detail_str("Venue", motec_log.venue_name)
    add_detail_str("Venue Category", "")
    add_detail_str("Driver", motec_log.driver)
    add_detail_str("Team", "")
    add_detail_str("Vehicle Id", motec_log.vehicle_id)
    add_detail_str("Vehicle Number", "")
    add_detail_str("Vehicle Desc", motec_log.vehicle_comment)
    add_detail_str("Engine Id", "")
    add_detail_str("Session", motec_log.event_session)
    add_detail_str("Start Lap", "")
    add_detail_str("Short Comment", motec_log.short_comment)
    add_detail_str("Long Comment", motec_log.long_comment)

    date_str = motec_log.datetime.strftime("%d/%m/%Y")
    time_str = motec_log.datetime.strftime("%H:%M:%S")
    ET.SubElement(details, "DateTime", {"Id": "Log Date", "Value": date_str})
    ET.SubElement(details, "DateTime", {"Id": "Log Time", "Value": time_str})

    add_detail_str("Sky", "")
    add_detail_str("Wind Direction", "")
    add_detail_str("Weather Comment", "")
    add_detail_str("Vehicle Type", motec_log.vehicle_type)
    add_detail_str("Vehicle Drive Type", "")
    add_detail_str("Vehicle Comment", motec_log.vehicle_comment)

    total_laps = laps_info.get("total_laps", 1) if laps_info else 1
    fastest_time_sec = laps_info.get("fastest_time", 0.0) if laps_info else 0.0

    if fastest_time_sec > 0:
        m, s = divmod(fastest_time_sec, 60)
        fastest_time_str = f"{int(m)}:{s:06.3f}"
    else:
        fastest_time_str = "0:00.000"

    add_detail_str("Total Laps", str(total_laps))
    add_detail_str("Fastest Time", fastest_time_str)

    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xml_str)
