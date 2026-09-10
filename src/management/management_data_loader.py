import json
from typing import Dict, Any

def load_reservoir_metadata(live_json_path: str, irrigation_live_json_path: str, historical_thresholds_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads threshold and capacity metadata from live JSON files, and historical inflow 
    thresholds from the precalculated training-data JSON.
    Returns a dictionary keyed by reservoir name (case-insensitive).
    """
    metadata = {}
    
    def parse_float(val):
        try:
            if val and str(val).strip() != "":
                return float(val)
        except ValueError:
            pass
        return None
        
    for path in [live_json_path, irrigation_live_json_path]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for dam in data.get('dams', []):
                    name = dam.get('name', '').lower().strip()
                    if not name:
                        continue
                    
                    metadata[name] = {
                        'name': dam.get('name'),
                        'FRL': parse_float(dam.get('FRL')),
                        'liveStorageAtFRL': parse_float(dam.get('liveStorageAtFRL')),
                        'blueLevel': parse_float(dam.get('blueLevel')),
                        'orangeLevel': parse_float(dam.get('orangeLevel')),
                        'redLevel': parse_float(dam.get('redLevel'))
                    }
        except FileNotFoundError:
            pass
            
    # Load historical inflow thresholds
    try:
        with open(historical_thresholds_path, 'r') as f:
            thresholds = json.load(f)
            for raw_name, threshold in thresholds.items():
                name = raw_name.lower().strip()
                if name in metadata:
                    metadata[name]['historical_95th_inflow'] = parse_float(threshold)
                else:
                    metadata[name] = {'historical_95th_inflow': parse_float(threshold)}
    except FileNotFoundError:
        pass
            
    return metadata

def get_current_state(historic_data_path: str) -> Dict[str, Any]:
    """
    Reads the latest telemetry state from a historic JSON file.
    """
    try:
        with open(historic_data_path, 'r') as f:
            data = json.load(f)
            records = data.get('data', [])
            if not records:
                return {}
            
            latest = records[0]
            
            def parse_float(val):
                try:
                    if val and str(val).strip() != "":
                        return float(val)
                except ValueError:
                    pass
                return None
                
            return {
                'date': latest.get('date'),
                'waterLevel': parse_float(latest.get('waterLevel')),
                'liveStorage': parse_float(latest.get('liveStorage')),
                'inflow': parse_float(latest.get('inflow'))
            }
    except FileNotFoundError:
        return {}
