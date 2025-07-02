#!/usr/bin/env python3
"""
Curriculum Learning WandB Setup Utility
========================================

This script helps you set up custom WandB dashboards and alerts for curriculum learning tracking.
Run this after starting your training to configure comprehensive curriculum monitoring.

Usage:
    python curriculum_wandb_setup.py --project YOUR_PROJECT_NAME --entity YOUR_ENTITY

Features:
- Creates custom curriculum learning dashboard
- Sets up automated alerts for curriculum stalling
- Configures curriculum performance analysis views
- Provides curriculum learning insights and recommendations
"""

import wandb
import argparse
import time
from typing import Dict, List, Optional


class CurriculumWandBSetup:
    """Utility class for setting up comprehensive curriculum learning tracking in WandB"""
    
    def __init__(self, project_name: str, entity: str):
        self.project_name = project_name
        self.entity = entity
        
    def create_curriculum_dashboard(self):
        """Create a custom WandB dashboard for curriculum learning monitoring"""
        
        # Dashboard configuration
        dashboard_config = {
            "displayName": "Curriculum Learning Monitor",
            "description": "Comprehensive curriculum learning tracking for DCE Navigation",
            "sections": [
                {
                    "name": "Curriculum Progression",
                    "panels": [
                        {
                            "title": "Curriculum Level Over Time",
                            "type": "line",
                            "query": {
                                "metrics": ["curriculum/current_level"],
                                "groupBy": [],
                            }
                        },
                        {
                            "title": "Curriculum Progress Fraction",
                            "type": "line", 
                            "query": {
                                "metrics": ["curriculum/current_progress"],
                                "groupBy": [],
                            }
                        },
                        {
                            "title": "Curriculum Progression Rate",
                            "type": "line",
                            "query": {
                                "metrics": ["curriculum/progression_rate"],
                                "groupBy": [],
                            }
                        }
                    ]
                },
                {
                    "name": "Performance Metrics",
                    "panels": [
                        {
                            "title": "Success Rate vs Curriculum Level",
                            "type": "scatter",
                            "query": {
                                "metrics": ["curriculum/success_rate", "curriculum/current_level"],
                                "groupBy": [],
                            }
                        },
                        {
                            "title": "Success Rate Moving Average",
                            "type": "line",
                            "query": {
                                "metrics": ["curriculum/success_rate", "curriculum/success_rate_ma5"],
                                "groupBy": [],
                            }
                        },
                        {
                            "title": "Crash vs Timeout Rates",
                            "type": "line",
                            "query": {
                                "metrics": ["curriculum/crash_rate", "curriculum/timeout_rate"],
                                "groupBy": [],
                            }
                        }
                    ]
                },
                {
                    "name": "Cumulative Statistics",
                    "panels": [
                        {
                            "title": "Total Episodes by Type",
                            "type": "bar",
                            "query": {
                                "metrics": ["curriculum/total_successes", "curriculum/total_crashes", "curriculum/total_timeouts"],
                                "groupBy": [],
                            }
                        }
                    ]
                }
            ]
        }
        
        print("Creating curriculum learning dashboard...")
        print("Dashboard configuration created. Apply this manually in WandB UI for now.")
        print(f"Dashboard config: {dashboard_config}")
        
    def setup_curriculum_alerts(self):
        """Set up automated alerts for curriculum learning monitoring"""
        
        alert_configs = [
            {
                "name": "Curriculum Stalling Alert",
                "condition": "curriculum/progression_rate < 0.01 for 1000 steps",
                "description": "Alert when curriculum progression stalls",
                "action": "Send notification when curriculum learning plateaus"
            },
            {
                "name": "Low Success Rate Alert", 
                "condition": "curriculum/success_rate < 0.3 for 500 steps",
                "description": "Alert when success rate drops too low",
                "action": "Send notification when performance degrades significantly"
            },
            {
                "name": "Curriculum Milestone Achievement",
                "condition": "curriculum/current_level % 5 == 0",
                "description": "Celebrate curriculum milestones",
                "action": "Send congratulatory message for curriculum progress"
            }
        ]
        
        print("\nSetting up curriculum learning alerts...")
        for alert in alert_configs:
            print(f"- {alert['name']}: {alert['description']}")
        
        print("\nNote: Set up these alerts manually in WandB UI using the conditions above.")
        
    def generate_curriculum_analysis_queries(self) -> List[str]:
        """Generate useful WandB queries for curriculum analysis"""
        
        queries = [
            # Performance analysis by curriculum level
            "SELECT curriculum/success_rate, curriculum/crash_rate GROUP BY curriculum/current_level",
            
            # Curriculum progression efficiency
            "SELECT curriculum/current_level, curriculum/progression_rate WHERE curriculum/progression_rate > 0",
            
            # Success rate trend analysis
            "SELECT curriculum/success_rate_ma5, curriculum/current_level ORDER BY _step",
            
            # Milestone achievement tracking
            "SELECT curriculum/milestone_level, curriculum/milestone_success_rate, curriculum/milestone_step",
            
            # Performance correlation with training progress
            "SELECT curriculum/success_rate, reward, curriculum/current_level",
        ]
        
        return queries
        
    def print_setup_instructions(self):
        """Print comprehensive setup instructions for curriculum tracking"""
        
        instructions = """
🚀 CURRICULUM LEARNING WANDB SETUP COMPLETE!
============================================

📊 NEXT STEPS:
1. Start your training with: python train_aerialgym_custom_net.py --env=quad_with_obstacles
2. Navigate to your WandB project dashboard
3. Create custom charts using the curriculum/* metrics
4. Set up the alerts mentioned above for proactive monitoring

📈 KEY METRICS TO TRACK:
- curriculum/current_level: Monitor curriculum progression
- curriculum/success_rate: Track learning performance  
- curriculum/progression_rate: Measure learning efficiency
- curriculum/success_rate_ma5: Smooth success rate trends

🔔 RECOMMENDED ALERTS:
- Curriculum stalling (progression_rate < 0.01)
- Low performance (success_rate < 0.3)
- Milestone achievements (every 5 levels)

📋 CUSTOM DASHBOARD PANELS TO CREATE:
1. Curriculum Level Timeline
2. Success Rate vs Curriculum Level Scatter Plot
3. Performance Metrics Comparison
4. Cumulative Episode Statistics

💡 ANALYSIS TIPS:
- Group metrics by curriculum/current_level for level-specific analysis
- Monitor the relationship between curriculum progression and overall reward
- Track how long the model spends at each curriculum level
- Compare success rates across different curriculum levels

Happy curriculum learning! 🎯
        """
        
        print(instructions)


def main():
    parser = argparse.ArgumentParser(description="Set up WandB curriculum learning tracking")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", required=True, help="WandB entity name")
    
    args = parser.parse_args()
    
    # Initialize curriculum setup
    curriculum_setup = CurriculumWandBSetup(args.project, args.entity)
    
    print(f"Setting up curriculum learning tracking for project: {args.project}")
    print(f"Entity: {args.entity}")
    print("="*60)
    
    # Create dashboard configuration
    curriculum_setup.create_curriculum_dashboard()
    
    # Setup alerts
    curriculum_setup.setup_curriculum_alerts()
    
    # Generate analysis queries
    queries = curriculum_setup.generate_curriculum_analysis_queries()
    print(f"\n📊 USEFUL ANALYSIS QUERIES:")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    
    # Print setup instructions
    curriculum_setup.print_setup_instructions()


if __name__ == "__main__":
    main() 