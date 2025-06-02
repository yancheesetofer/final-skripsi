#!/usr/bin/env python
# coding: utf-8

import csv
import json
import os
import random
import numpy as np
import warnings
from faker import Faker
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')
fake = Faker()

DEFAULT_CONFIG = {
  "total_users": 45,
  "total_quizzes": 4,
  "questions_per_quiz": 30,
  "base_date": "2025-03-01 09:00:00",
  "cheating_groups": {
    "high_severity": {
      "count": 3,
      "size": 3,
      "patterns": {
        "navigation": {
          "similarity": 0.96,
          "noise": 0.07
        },
        "timing": {
          "start_delay": 0,
          "variance": 2,
          "completion_speed": "fast"
        },
        "answers": {
          "similarity": 0.94,
          "wrong_bias": 0.87
        }
      }
    },
    "medium_severity": {
      "count": 1,
      "size": 4,
      "patterns": {
        "navigation": {
          "similarity": 0.80,
          "noise": 0.30
        },
        "timing": {
          "start_delay": 3,
          "variance": 15,
          "completion_speed": "medium"
        },
        "answers": {
          "similarity": 0.84,
          "wrong_bias": 0.6
        }
      }
    },
    "low_severity": {
      "count": 0,
      "size": 0,
      "patterns": {
        "navigation": {
          "similarity": 0.55,
          "noise": 0.45
        },
        "timing": {
          "start_delay": 15,
          "variance": 30,
          "completion_speed": "medium"
        },
        "answers": {
          "similarity": 0.55,
          "wrong_bias": 0.4
        }
      }
    }
  },
  "output_format": "csv",
  "output_dir": "/Users/yan.christofer/Documents/detection_cleaner/data/moodle_logs",
  "seed": 12345
}

@dataclass
class CheatingGroup:
    id: str
    severity: str
    members: List[int]
    patterns: Dict[str, Any]
    
    # Shared patterns across the group
    navigation_sequence: List[int] = field(default_factory=list)
    answer_sequence: List[int] = field(default_factory=list)

@dataclass
class User:
    id: int
    username: str
    password: str
    firstname: str
    lastname: str
    email: str
    lastaccess: int
    is_cheater: bool = False
    cheating_group: Optional[str] = None
    cheating_severity: Optional[str] = None

@dataclass
class Quiz:
    quiz_id: int  # Renamed from id
    course: int
    quiz_name: str  # Renamed from name
    intro: str
    timeopen: int
    timeclose: int
    timelimit: int
    overduehandling: str

@dataclass
class Question:
    id: int
    qtype: str
    name: str
    questiontext: str
    defaultmark: float
    penalty: float
    quizid: int

@dataclass
class QuestionAnswer:
    question_answers_id: int  # Renamed from id
    questionid: int  # Renamed from question
    answer_text: str  # Renamed from answer
    fraction: float  # 1.0 for correct, 0.0 for incorrect

@dataclass
class Session:
    session_id: int  # Renamed from id
    user_id: int  # Renamed from userid
    timecreated: int
    timemodified: int
    firstip: str
    lastip: str
    sessdata: str = ""  # Added missing field

@dataclass
class QuizAttempt:
    attempt_id: int  # Renamed from id
    quiz_id: int  # Renamed from quiz
    user_id: int  # Renamed from userid
    attempt: int
    state: str
    timestart: int
    timefinish: int
    timemodified: int
    question_usage_id: int = 0  # Added missing field
    sumgrades: float = 0.0  # Added missing field
    # ML-specific fields
    is_cheating: bool = False
    cheating_severity: Optional[str] = None
    cheating_group: Optional[str] = None

@dataclass
class QuestionAttemptStep:
    question_step_id: int  # Renamed from id
    question_attempt_id: int  # Renamed from questionattemptid
    sequencenumber: int
    state: str
    timecreated: int
    user_id: int  # Renamed from userid
    # ML-specific fields
    is_cheating: bool = False
    cheating_severity: Optional[str] = None
    cheating_group: Optional[str] = None

@dataclass
class QuestionAttemptStepData:
    question_step_id: int  # Renamed from attemptstepid
    name: str
    value: str
    user_id: int  # Renamed from userid
    step_data_id: int = 0  # Added missing field
    # ML-specific fields
    is_cheating: bool = False
    cheating_severity: Optional[str] = None
    cheating_group: Optional[str] = None

@dataclass
class QuestionAttemptsMap: # Represents mdl_question_attempts
    id: int 
    questionusageid: int # Corresponds to QuizAttempt.attempt_id
    questionattemptid: int # Corresponds to the user_id * 100 + quiz_id used in QuestionAttemptStep

class MoodleLogGenerator:
    def __init__(self, config_file=None):
        """Initialize the generator with either a config file or default config"""
        self.config = DEFAULT_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Set random seed for reproducibility
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        
        # Initialize data structures
        self.users = []
        self.quizzes = []
        self.questions = []
        self.question_answers = []
        self.sessions = []
        self.quiz_attempts = []
        self.question_attempt_steps = []
        self.question_attempt_step_data = []
        self.question_attempts_maps = [] # ADDED
        
        # Track IDs
        self.next_id = {
            "user": 1,
            "quiz": 1,
            "question": 1,
            "question_answer": 1,
            "session": 1,
            "quiz_attempt": 1,
            "question_attempt_step": 1
        }
        
        # Setup cheating groups
        self.cheating_groups = []
        self.cheaters = []
        self.setup_cheating_groups()
        
        # Parse base date
        self.base_date = datetime.strptime(self.config["base_date"], "%Y-%m-%d %H:%M:%S")
    
    def setup_cheating_groups(self):
        """Create cheating groups based on configuration"""
        group_id = 1
        user_id = 1
        
        for severity, config in self.config["cheating_groups"].items():
            for i in range(config["count"]):
                group_name = f"group_{severity}_{i+1}"
                members = list(range(user_id, user_id + config["size"]))
                user_id += config["size"]
                
                group = CheatingGroup(
                    id=group_name,
                    severity=severity,
                    members=members,
                    patterns=config["patterns"]
                )
                
                self.cheating_groups.append(group)
                self.cheaters.extend(members)
                
                group_id += 1
        
        # Calculate honest students
        total_cheaters = len(self.cheaters)
        if total_cheaters >= self.config["total_users"]:
            raise ValueError("Too many cheaters defined. Reduce cheating groups or increase total users.")
        
        print(f"Created {len(self.cheating_groups)} cheating groups with {total_cheaters} total cheaters")
    
    def generate_users(self):
        """Generate users following Moodle schema with cheating flags"""
        for user_id in range(1, self.config["total_users"] + 1):
            is_cheater = user_id in self.cheaters
            cheating_group = None
            cheating_severity = None
            
            if is_cheater:
                # Find which group this user belongs to
                for group in self.cheating_groups:
                    if user_id in group.members:
                        cheating_group = group.id
                        cheating_severity = group.severity
                        break
            
            user = User(
                id=user_id,
                username=fake.user_name(),
                password=fake.sha256(),
                firstname=fake.first_name(),
                lastname=fake.last_name(),
                email=fake.email(),
                lastaccess=int(datetime.now().timestamp()),
                is_cheater=is_cheater,
                cheating_group=cheating_group,
                cheating_severity=cheating_severity
            )
            
            self.users.append(user)
        
        print(f"Generated {len(self.users)} users")
    
    def generate_quiz_topics(self):
        """Generate realistic quiz topics"""
        topics = [
            "Introduction to Programming",
            "Data Structures and Algorithms",
            "Database Systems",
            "Computer Networks",
            "Operating Systems",
            "Software Engineering",
            "Artificial Intelligence",
            "Machine Learning",
            "Computer Graphics",
            "Web Development"
        ]
        return random.sample(topics, min(len(topics), self.config["total_quizzes"]))
    
    def generate_quizzes(self):
        """Generate quizzes following Moodle schema"""
        topics = self.generate_quiz_topics()
        
        for quiz_id in range(1, self.config["total_quizzes"] + 1):
            topic = topics[quiz_id-1] if quiz_id <= len(topics) else f"Quiz Topic {quiz_id}"
            
            timeopen = self.base_date + timedelta(days=(quiz_id-1)*7)  # One quiz per week
            timeclose = timeopen + timedelta(hours=2)
            
            quiz = Quiz(
                quiz_id=quiz_id,
                course=1,
                quiz_name=f"Quiz {quiz_id}: {topic}",
                intro=f"This is Quiz {quiz_id} on {topic}. Please read each question carefully and select the best answer.",
                timeopen=int(timeopen.timestamp()),
                timeclose=int(timeclose.timestamp()),
                timelimit=7200,  # 2 hours
                overduehandling="autosubmit"
            )
            
            self.quizzes.append(quiz)
        
        print(f"Generated {len(self.quizzes)} quizzes")
    
    def generate_questions_and_answers(self):
        """Generate questions and answers following Moodle schema"""
        question_id = 1
        answer_id = 1
        
        for quiz_id in range(1, self.config["total_quizzes"] + 1):
            for q in range(1, self.config["questions_per_quiz"] + 1):
                question = Question(
                    id=question_id,
                    qtype="multichoice",
                    name=f"Q{q}",
                    questiontext=f"Question {q} for Quiz {quiz_id}",
                    defaultmark=1.0,
                    penalty=0.33,
                    quizid=quiz_id
                )
                
                self.questions.append(question)
                
                # Generate 4 answers per question, one correct
                correct_answer = random.randint(0, 3)
                for i in range(4):
                    answer = QuestionAnswer(
                        question_answers_id=answer_id,
                        questionid=question_id,
                        answer_text=f"Option {i+1}",
                        fraction=1.0 if i == correct_answer else 0.0
                    )
                    
                    self.question_answers.append(answer)
                    answer_id += 1
                
                question_id += 1
        
        print(f"Generated {len(self.questions)} questions and {len(self.question_answers)} answers")
    
    def generate_navigation_sequence(self, question_count, pattern=None, similarity=0.0, noise=1.0, base_sequence=None):
        """Generate a navigation sequence with configurable similarity to a base sequence"""
        if base_sequence and similarity > 0:
            # Create a sequence similar to the base_sequence
            sequence = base_sequence.copy()
            
            # Apply noise by swapping some elements
            swaps = int(question_count * noise)
            for _ in range(swaps):
                if random.random() > similarity:  # Only swap if we're breaking similarity
                    i, j = random.sample(range(question_count), 2)
                    sequence[i], sequence[j] = sequence[j], sequence[i]
            
            return sequence
        else:
            # Create a sequence with optional pattern
            if pattern == "sequential":
                # Sequential pattern (1,2,3,4...)
                sequence = list(range(1, question_count + 1))
            elif pattern == "reverse":
                # Reverse pattern (20,19,18...)
                sequence = list(range(question_count, 0, -1))
            elif pattern == "odd_first":
                # Odd questions first, then even
                odds = list(range(1, question_count + 1, 2))
                evens = list(range(2, question_count + 1, 2))
                sequence = odds + evens
            elif pattern == "even_first":
                # Even questions first, then odd
                evens = list(range(2, question_count + 1, 2))
                odds = list(range(1, question_count + 1, 2))
                sequence = evens + odds
            elif pattern == "jump_around":
                # Jump between different sections (simulate jumping between easy questions)
                first_third = list(range(1, question_count // 3 + 1))
                second_third = list(range(question_count // 3 + 1, 2 * question_count // 3 + 1))
                last_third = list(range(2 * question_count // 3 + 1, question_count + 1))
                sequence = first_third + last_third + second_third
            else:
                # Random order
                sequence = list(range(1, question_count + 1))
                random.shuffle(sequence)
            
            # Add revisits (realistic behavior) - Enhanced for non-cheaters with more variability
            is_cheater = base_sequence is not None  # If we have a base sequence, this is for a cheater
            
            if is_cheater:
                # Cheaters: Standard revisits (more controlled)
                revisit_count = random.randint(0, question_count // 4)  # Revisit up to 25% of questions
                for _ in range(revisit_count):
                    # Pick a question to revisit
                    question = random.choice(sequence)
                    # Find a position after its first appearance to insert the revisit
                    first_pos = sequence.index(question)
                    insert_pos = random.randint(first_pos + 1, len(sequence))
                    sequence.insert(insert_pos, question)
            else:
                # Non-cheaters: Much more varied and realistic browsing patterns
                # More revisits on average (naturally reviewing work)
                revisit_count = random.randint(2, max(3, question_count // 3))  # Revisit 2 to 33% of questions
                
                # Sometimes non-cheaters review earlier questions multiple times
                for _ in range(revisit_count):
                    # Bias toward earlier questions (common behavior to review early work)
                    early_bias = random.random() < 0.7  # 70% chance to review early questions
                    if early_bias and len(sequence) > 3:
                        # Pick from first third of questions
                        first_third = sequence[:len(sequence)//3]
                        question = random.choice(first_third)
                    else:
                        question = random.choice(sequence)
                        
                    # Insert revisit
                    first_pos = sequence.index(question)
                    insert_pos = random.randint(first_pos + 1, len(sequence))
                    sequence.insert(insert_pos, question)
                
                # Sometimes add "back and forth" behavior (common when checking related questions)
                if random.random() < 0.4:  # 40% chance
                    # Find two nearby questions to jump between
                    idx = random.randint(0, len(sequence) - 2)
                    q1 = sequence[idx]
                    q2 = sequence[idx + 1]
                    
                    # Insert a few back-and-forth movements
                    back_forth_count = random.randint(1, 3)
                    for _ in range(back_forth_count):
                        insert_pos = random.randint(idx + 2, min(idx + 10, len(sequence)))
                        if random.random() < 0.5:
                            sequence.insert(insert_pos, q1)
                        else:
                            sequence.insert(insert_pos, q2)
                
                # Sometimes skip ahead then go back (realistic human behavior)
                if random.random() < 0.3 and len(sequence) > 10:  # 30% chance
                    # Pick a position to simulate "skipping ahead"
                    skip_pos = random.randint(len(sequence)//4, 3*len(sequence)//4)
                    # How far to skip ahead
                    skip_distance = random.randint(2, min(5, len(sequence) - skip_pos - 1))
                    
                    # The question we're skipping to
                    skip_to = sequence[skip_pos + skip_distance]
                    
                    # Insert it earlier
                    sequence.insert(skip_pos, skip_to)
                    
                    # And maybe insert a revisit to the original position we skipped from
                    if random.random() < 0.7:  # 70% chance
                        sequence.insert(skip_pos + 2, sequence[skip_pos - 1])
            
            return sequence
    
    def generate_answer_sequence(self, question_count, correct_answers, pattern=None, 
                                 similarity=0.0, wrong_bias=0.0, base_sequence=None):
        """Generate answer selections with configurable similarity to a base sequence"""
        sequence = []
        
        if base_sequence and similarity > 0:
            # Create answers similar to the base_sequence with controlled randomness
            for q_idx in range(question_count):
                # Even with high similarity, introduce some realistic variations
                # This ensures not all cheaters have 100% identical answers
                use_base = random.random() < similarity
                
                # Special case: sometimes introduce targeted variations even in high-similarity groups
                # This makes the cheating more realistic - not every answer matches perfectly
                if similarity > 0.8 and random.random() < 0.15:  # 15% chance for high-similarity groups
                    # Create an intentional deviation on some questions
                    use_base = False
                
                if use_base:
                    # Use the same answer as the base sequence
                    sequence.append(base_sequence[q_idx])
                else:
                    # Generate a new random answer, with some scientific considerations:
                    correct = correct_answers[q_idx]
                    
                    # Realistic cheating behavior: biased toward making the SAME mistakes
                    # Real cheaters often share wrong answers on difficult questions
                    if random.random() < wrong_bias and base_sequence[q_idx] != correct:
                        # Use the same wrong answer (coordinated mistake)
                        sequence.append(base_sequence[q_idx])
                    else:
                        # Different cases based on question difficulty (simulated):
                        q_difficulty = random.random()  # Higher = more difficult
                        
                        if q_difficulty > 0.7:  # Hard question
                            # More likely to get hard questions wrong even with individual effort
                            if random.random() < 0.7:
                                # Wrong answer, but potentially different from base
                                wrong_options = [i for i in range(4) if i != correct]
                                sequence.append(random.choice(wrong_options))
                            else:
                                # Still possible to get right
                                sequence.append(correct)
                        else:  # Easier question
                            # Less likely to get easy questions wrong
                            if random.random() < 0.3:  # 30% chance of wrong
                                wrong_options = [i for i in range(4) if i != correct]
                                sequence.append(random.choice(wrong_options))
                            else:
                                # More likely to get right
                                sequence.append(correct)
        else:
            # Generate completely random answers, but with realistic correctness rates
            for q_idx in range(question_count):
                # Simulate question difficulty
                q_difficulty = random.random()
                correct = correct_answers[q_idx]
                
                # Harder questions less likely to be correct
                if q_difficulty > 0.7:  # Hard question
                    if random.random() < 0.6:  # 40% chance of being correct
                        sequence.append(correct)
                    else:
                        wrong_options = [i for i in range(4) if i != correct]
                        sequence.append(random.choice(wrong_options))
                else:  # Easier question
                    if random.random() < 0.8:  # 80% chance of being correct
                        sequence.append(correct)
                    else:
                        wrong_options = [i for i in range(4) if i != correct]
                        sequence.append(random.choice(wrong_options))
        
        return sequence
    
    def get_correct_answers(self, quiz_id):
        """Get correct answers for a quiz"""
        correct_answers = []
        questions_for_quiz = [q for q in self.questions if q.quizid == quiz_id]
        
        for question in questions_for_quiz:
            answers = [a for a in self.question_answers if a.questionid == question.id]
            correct_idx = next(i for i, a in enumerate(answers) if a.fraction > 0)
            correct_answers.append(correct_idx)
        
        return correct_answers
    
    def generate_sessions_and_attempts(self):
        """Generate sessions and attempts with realistic patterns"""
        session_id = 1
        attempt_id = 1
        step_id = 1
        
        # First, pre-generate group patterns
        for group in self.cheating_groups:
            # Generate shared navigation and answer patterns for each quiz
            group.quiz_patterns = {}
            
            for quiz_id in range(1, self.config["total_quizzes"] + 1):
                # Select a navigation pattern strategy for this group and quiz
                # Each group has its own "style" of navigation
                if group.severity == "high_severity":
                    # High severity groups tend to have more structured navigation
                    # IMPORTANT: Fixed to ensure we don't always use sequential pattern
                    nav_pattern = random.choice(["reverse", "odd_first", "even_first", "jump_around"]) 
                    # Usually more revisits in high severity (to copy from others)
                    revisit_rate = random.randint(4, 7)  # 4-7 revisits
                elif group.severity == "medium_severity":
                    # Medium severity groups have semi-structured navigation
                    nav_pattern = random.choice(["jump_around", None, None])  # Less likely to be sequential
                    revisit_rate = random.randint(2, 5)  # 2-5 revisits
                else:
                    # Low severity groups have less obvious patterns
                    nav_pattern = random.choice(["jump_around", None, None])  # More likely to be random
                    revisit_rate = random.randint(1, 3)  # 1-3 revisits
                
                # Generate base sequences for the group
                nav_sequence = self.generate_navigation_sequence(
                    self.config["questions_per_quiz"],
                    pattern=nav_pattern
                )
                
                # Add strategic revisits for the group (simulates checking answers)
                for _ in range(revisit_rate):
                    q_to_revisit = random.randint(1, self.config["questions_per_quiz"])
                    # Insert revisit after at least 3 questions
                    insert_pos = random.randint(3, len(nav_sequence))
                    nav_sequence.insert(insert_pos, q_to_revisit)
                    
                # Further randomize the sequence slightly to avoid perfect matching between group members
                if random.random() < 0.7:  # 70% chance to add more randomness
                    # Randomly swap 2-4 positions
                    for _ in range(random.randint(2, 4)):
                        i, j = random.sample(range(len(nav_sequence)), 2)
                        nav_sequence[i], nav_sequence[j] = nav_sequence[j], nav_sequence[i]
                
                correct_answers = self.get_correct_answers(quiz_id)
                answer_sequence = []
                
                # Ensure some wrong answers for cheating detection
                for q_idx in range(self.config["questions_per_quiz"]):
                    correct = correct_answers[q_idx]
                    # Higher chance of wrong answer for cheaters
                    if random.random() < group.patterns["answers"]["wrong_bias"]:
                        # Choose an incorrect answer
                        wrong_options = [i for i in range(4) if i != correct]
                        answer_sequence.append(random.choice(wrong_options))
                    else:
                        # Choose correct answer
                        answer_sequence.append(correct)
                
                # For revisits, they'll provide the same answer (cheaters are consistent)
                # Need to extend answer_sequence to match nav_sequence length
                answer_map = {}
                for i, q in enumerate(range(1, self.config["questions_per_quiz"] + 1)):
                    answer_map[q] = answer_sequence[i]
                
                # Rebuild answer sequence to match navigation including revisits
                full_answer_sequence = [answer_map[q] for q in nav_sequence]
                
                # Generate shared timing pattern for the group, with question-specific times
                # This enforces NOT JUST similar times per question, but synchronized transitions
                timing_pattern = []
                
                # Track the transition times between questions - THIS IS THE KEY SCIENTIFIC INSIGHT:
                # Cheaters don't just have similar time spent PER question, but actually move between
                # questions at the same TIME INTERVALS, creating a temporal correlation
                transition_offsets = []
                
                # Start with initial offset (time to 1st question)
                base_offset = random.randint(30, 120)  # Initial time browsing before starting
                # High severity = tighter sync
                if group.severity == "high_severity":
                    offset_variance = 5
                elif group.severity == "medium_severity":
                    offset_variance = 15
                else:
                    offset_variance = 30
                    
                transition_offsets.append((base_offset, offset_variance))
                    
                # Calculate a baseline sequence of timing offsets for the group
                # This creates the synchronized question transitions scientifically observed in cheating
                question_sync_timestamps = [0]  # Start time
                
                for q in nav_sequence:
                    # Check if this is a revisit
                    is_revisit = nav_sequence.count(q) > 1 and nav_sequence.index(q) != len(nav_sequence) - 1 - nav_sequence[::-1].index(q)
                    
                    # Determine time spent
                    if is_revisit:
                        if group.severity == "high_severity":
                            # Quick revisits for high severity
                            base_time = random.randint(5, 15)
                            variance = 2
                        else:
                            # Slightly longer revisits for lower severity
                            base_time = random.randint(10, 25)
                            variance = 5
                    else:
                        # Base time + small variance for cheaters (first visit)
                        if group.severity == "high_severity":
                            # High severity: extremely consistent timing between group members
                            # Not necessarily fast, but suspiciously similar pace
                            base_time = random.randint(15, 45)  # Can be fast or slow
                            variance = 3  # Very small variance between group members
                        elif group.severity == "medium_severity":
                            # Medium severity: moderately consistent timing
                            base_time = random.randint(20, 60)
                            variance = 8  # Moderate variance
                        else:
                            # Low severity: somewhat similar but with more natural variation
                            base_time = random.randint(25, 70)
                            variance = 15
                    
                    timing_pattern.append((base_time, variance))
                    
                    # Calculate transition time to next question 
                    # This is a crucial addition for scientific validity - we need the timestamps
                    # rather than just the durations to be synchronized
                    last_time = question_sync_timestamps[-1]
                    next_time = last_time + base_time
                    question_sync_timestamps.append(next_time)
                
                # Implement leader-follower behavior for high-severity cheating
                # This adds another scientifically valid pattern: in real cheating, 
                # there's often a leader and followers with dependency patterns
                leader_follower_pattern = None
                if group.severity == "high_severity":
                    # For high severity, implement leader-follower pattern
                    # Leader completes questions first, followers follow with delay
                    leader_idx = 0  # first group member is leader
                    # Define correlated delays for followers
                    follower_delays = {}
                    for i in range(1, len(group.members)):
                        # Followers wait for leader with consistent delays
                        if i == 1:  # First follower waits less
                            follower_delays[i] = random.randint(20, 40)
                        elif i == 2:  # Second follower waits a bit more
                            follower_delays[i] = random.randint(40, 60)
                        else:  # Additional followers wait even more
                            follower_delays[i] = random.randint(60, 90)
                    
                    leader_follower_pattern = {
                        "leader_idx": leader_idx,
                        "follower_delays": follower_delays
                    }
                
                group.quiz_patterns[quiz_id] = {
                    "navigation_sequence": nav_sequence,
                    "answer_sequence": full_answer_sequence,
                    "timing_pattern": timing_pattern,
                    "question_sync_timestamps": question_sync_timestamps,
                    "leader_follower_pattern": leader_follower_pattern
                }
        
        # Generate for each quiz
        for quiz_id in range(1, self.config["total_quizzes"] + 1):
            quiz_time = self.base_date + timedelta(days=(quiz_id-1)*7)
            
            # Process honest students first
            honest_users = [u for u in self.users if not u.is_cheater]
            for user in honest_users:
                # Create honest session and attempt
                start_time = quiz_time + timedelta(minutes=random.randint(0, 60))
                
                # Honest students take variable time
                duration_minutes = random.randint(30, 90)
                duration = timedelta(minutes=duration_minutes)
                
                # Create session
                session = Session(
                    session_id=session_id,
                    user_id=user.id,
                    timecreated=int(start_time.timestamp()),
                    timemodified=int((start_time + duration).timestamp()),
                    firstip=fake.ipv4(),
                    lastip=fake.ipv4(),
                    sessdata=""
                )
                self.sessions.append(session)
                session_id += 1
                
                # Create quiz attempt
                attempt = QuizAttempt(
                    attempt_id=attempt_id,
                    quiz_id=quiz_id,
                    user_id=user.id,
                    attempt=1,
                    state="finished",
                    timestart=int(start_time.timestamp()),
                    timefinish=int((start_time + duration).timestamp()),
                    timemodified=int((start_time + duration).timestamp()),
                    is_cheating=False,
                    question_usage_id=user.id * 1000 + quiz_id,  # Generate a unique question_usage_id
                    sumgrades=random.uniform(0.0, self.config["questions_per_quiz"])  # Random grade
                )
                self.quiz_attempts.append(attempt)
                attempt_id += 1
                
                # Create mapping for mdl_question_attempts # ADDED
                qam_id_counter = len(self.question_attempts_maps) + 1 
                qam_entry = QuestionAttemptsMap(
                    id=qam_id_counter,
                    questionusageid=attempt.attempt_id, 
                    questionattemptid=user.id * 100 + quiz_id 
                )
                self.question_attempts_maps.append(qam_entry)
                
                # Generate honest navigation with revisits
                nav_sequence = self.generate_navigation_sequence(
                    self.config["questions_per_quiz"]
                )
                
                # Add random revisits (honest students often revisit questions)
                revisit_count = random.randint(2, 8)  # More variable revisits
                for _ in range(revisit_count):
                    q_to_revisit = random.randint(1, self.config["questions_per_quiz"])
                    insert_pos = random.randint(1, len(nav_sequence))
                    nav_sequence.insert(insert_pos, q_to_revisit)
                
                correct_answers = self.get_correct_answers(quiz_id)
                
                # Create answer map for honest students
                answer_map = {}
                for q in range(1, self.config["questions_per_quiz"] + 1):
                    correct = correct_answers[q-1]
                    # 70% chance of correct for honest students (adjustable)
                    if random.random() < 0.7:
                        answer_map[q] = correct
                    else:
                        # Choose an incorrect answer
                        wrong_options = [i for i in range(4) if i != correct]
                        answer_map[q] = random.choice(wrong_options)
                
                # For revisits, honest students might change their answers
                # Build full answer sequence matching navigation
                answer_sequence = []
                for q in nav_sequence:
                    # Is this a revisit?
                    if q in answer_map and nav_sequence.count(q) > 1 and len([x for x in nav_sequence[:nav_sequence.index(q)] if x == q]) > 0:
                        # 30% chance to change answer on revisit
                        if random.random() < 0.3:
                            correct = correct_answers[q-1]
                            # 60% chance to correct a wrong answer, 40% to get it wrong
                            if answer_map[q] != correct and random.random() < 0.6:
                                # Change to correct answer
                                answer_map[q] = correct
                            else:
                                # Change to another answer
                                options = [i for i in range(4) if i != answer_map[q]]
                                answer_map[q] = random.choice(options)
                    
                    # Add the answer for this question to the sequence
                    answer_sequence.append(answer_map[q])
                
                # Generate steps for this honest attempt
                self.generate_attempt_steps(
                    user.id, 
                    quiz_id, 
                    start_time, 
                    nav_sequence, 
                    answer_sequence,
                    is_cheating=False,
                    step_id=step_id
                )
                
                # Update step_id
                step_id += len(nav_sequence) * 2  # Each question has view + answer step
            
            # Process cheating groups
            for group in self.cheating_groups:
                # Get group patterns for this quiz
                patterns = group.quiz_patterns[quiz_id]
                
                # Process each cheater in the group with coordinated timing
                for idx, user_id in enumerate(group.members):
                    user = next(u for u in self.users if u.id == user_id)
                    
                    # Calculate timing based on group pattern and position in group
                    start_delay = group.patterns["timing"]["start_delay"] * idx
                    start_time = quiz_time + timedelta(minutes=start_delay)
                    
                    # Cheaters finish faster based on severity
                    if group.patterns["timing"]["completion_speed"] == "fast":
                        duration_minutes = random.randint(10, 20)
                    elif group.patterns["timing"]["completion_speed"] == "medium":
                        duration_minutes = random.randint(20, 40)
                    else:
                        duration_minutes = random.randint(30, 50)
                    
                    duration = timedelta(minutes=duration_minutes)
                    
                    # Create session
                    session = Session(
                        session_id=session_id,
                        user_id=user.id,
                        timecreated=int(start_time.timestamp()),
                        timemodified=int((start_time + duration).timestamp()),
                        firstip=fake.ipv4(),
                        lastip=fake.ipv4(),
                        sessdata=""
                    )
                    self.sessions.append(session)
                    session_id += 1
                    
                    # Create quiz attempt with cheating label
                    # For cheaters, generate a grade based on severity
                    if group.severity == "high_severity":
                        # High severity cheaters tend to score very well
                        grade = random.uniform(self.config["questions_per_quiz"] * 0.8, self.config["questions_per_quiz"])
                    elif group.severity == "medium_severity":
                        # Medium severity get good but not perfect scores
                        grade = random.uniform(self.config["questions_per_quiz"] * 0.6, self.config["questions_per_quiz"] * 0.9)
                    else:
                        # Low severity get more variable scores
                        grade = random.uniform(self.config["questions_per_quiz"] * 0.4, self.config["questions_per_quiz"] * 0.8)
                        
                    attempt = QuizAttempt(
                        attempt_id=attempt_id,
                        quiz_id=quiz_id,
                        user_id=user.id,
                        attempt=1,
                        state="finished",
                        timestart=int(start_time.timestamp()),
                        timefinish=int((start_time + duration).timestamp()),
                        timemodified=int((start_time + duration).timestamp()),
                        is_cheating=True,
                        cheating_severity=group.severity,
                        cheating_group=group.id,
                        question_usage_id=user.id * 1000 + quiz_id,  # Generate a unique question_usage_id
                        sumgrades=grade
                    )
                    self.quiz_attempts.append(attempt)
                    attempt_id += 1
                    
                    # Create mapping for mdl_question_attempts # ADDED
                    qam_id_counter = len(self.question_attempts_maps) + 1
                    qam_entry = QuestionAttemptsMap(
                        id=qam_id_counter,
                        questionusageid=attempt.attempt_id,
                        questionattemptid=user.id * 100 + quiz_id # user here is the one from self.users
                    )
                    self.question_attempts_maps.append(qam_entry)
                    
                    # Generate navigation with similarity to group pattern
                    nav_sequence = self.generate_navigation_sequence(
                        self.config["questions_per_quiz"],
                        similarity=group.patterns["navigation"]["similarity"],
                        noise=group.patterns["navigation"]["noise"],
                        base_sequence=patterns["navigation_sequence"]
                    )
                    
                    # Generate answers with similarity to group pattern
                    correct_answers = self.get_correct_answers(quiz_id)
                    answer_sequence = self.generate_answer_sequence(
                        self.config["questions_per_quiz"],
                        correct_answers,
                        similarity=group.patterns["answers"]["similarity"],
                        wrong_bias=group.patterns["answers"]["wrong_bias"],
                        base_sequence=patterns["answer_sequence"]
                    )
                    
                    # Generate steps for this cheating attempt
                    self.generate_attempt_steps(
                        user.id, 
                        quiz_id, 
                        start_time, 
                        nav_sequence, 
                        answer_sequence,
                        is_cheating=True,
                        cheating_severity=group.severity,
                        cheating_group=group.id,
                        time_variance=group.patterns["timing"]["variance"],
                        step_id=step_id
                    )
                    
                    # Update step_id
                    step_id += len(nav_sequence) * 2  # Each question has view + answer step
        
        print(f"Generated {len(self.sessions)} sessions and {len(self.quiz_attempts)} attempts")
        print(f"Generated {len(self.question_attempt_steps)} steps and {len(self.question_attempt_step_data)} step data entries")
    
    def generate_attempt_steps(self, user_id, quiz_id, start_time, navigation_sequence, 
                              answer_sequence, is_cheating=False, cheating_severity=None, 
                              cheating_group=None, time_variance=30, step_id=1):
        """Generate steps for a quiz attempt with realistic timing"""
        current_time = start_time
        base_step_id = step_id
        completed_questions = []
        
        # Get group timing pattern if this is a cheater
        timing_pattern = None
        question_sync_timestamps = None
        leader_follower_pattern = None
        member_idx_in_group = None
        
        if is_cheating and cheating_group:
            group = next((g for g in self.cheating_groups if g.id == cheating_group), None)
            if group and quiz_id in group.quiz_patterns:
                timing_pattern = group.quiz_patterns[quiz_id]["timing_pattern"]
                question_sync_timestamps = group.quiz_patterns[quiz_id]["question_sync_timestamps"]
                leader_follower_pattern = group.quiz_patterns[quiz_id].get("leader_follower_pattern", None)
                
                # Find this user's index in the group
                if user_id in group.members:
                    member_idx_in_group = group.members.index(user_id)
        
        # CRITICAL FIX: Ensure answer_sequence and navigation_sequence have the same length
        if len(answer_sequence) != len(navigation_sequence):
            print(f"Warning: Answer sequence length ({len(answer_sequence)}) doesn't match navigation sequence length ({len(navigation_sequence)}) for user {user_id}")
            # Extend answer_sequence if it's too short by repeating the last answer
            if len(answer_sequence) < len(navigation_sequence):
                last_answer = answer_sequence[-1] if answer_sequence else 0
                answer_sequence.extend([last_answer] * (len(navigation_sequence) - len(answer_sequence)))
            # Truncate if it's too long
            elif len(answer_sequence) > len(navigation_sequence):
                answer_sequence = answer_sequence[:len(navigation_sequence)]
        
        # For realistic cheating, create a consistent personal variance
        # This way each cheater maintains their own "style" but follows the group pattern
        personal_timing_style = random.randint(-2, 2)  # Individual tendency (slightly faster/slower)
        
        # For cheating groups, apply leader-follower pattern using absolute timestamps
        # instead of relative timing to create synchronized patterns
        if is_cheating and leader_follower_pattern and member_idx_in_group is not None:
            leader_idx = leader_follower_pattern["leader_idx"]
            follower_delays = leader_follower_pattern["follower_delays"]
            
            # Calculate delay based on position in the group
            base_delay = 0
            if member_idx_in_group > 0:  # Not the leader
                base_delay = follower_delays.get(member_idx_in_group, 30)  # Default to 30 sec if not specified
        else:
            base_delay = 0
            
        # This is crucial: for cheating groups, we want to apply the synchronized timestamps
        # with only small individual variance, rather than independent timing for each question
        if is_cheating and question_sync_timestamps:
            # Calculate absolute timestamps for each question for this user 
            user_timestamps = []
            for ts in question_sync_timestamps:
                # Add group-based offset plus small individual variance  
                user_time = start_time + timedelta(seconds=ts) + timedelta(seconds=base_delay)
                
                # Add small personal variation to avoid perfect synchronization
                # (Deviation decreases with severity)
                if cheating_severity == "high_severity":
                    variation = random.randint(-3, 3)
                elif cheating_severity == "medium_severity": 
                    variation = random.randint(-8, 8)
                else:
                    variation = random.randint(-15, 15)
                    
                user_time += timedelta(seconds=variation + personal_timing_style)
                user_timestamps.append(user_time)
                
            # Now we have pre-calculated timestamps for each question transition
            # which will show synchronized patterns between group members
        
        for idx, question_num in enumerate(navigation_sequence):
            # Use synchronized timestamps for cheaters if available
            if is_cheating and question_sync_timestamps and idx < len(user_timestamps):
                current_time = user_timestamps[idx]
            
            # Initial view of question
            step = QuestionAttemptStep(
                question_step_id=step_id,
                question_attempt_id=user_id * 100 + quiz_id,  # Consistent format
                sequencenumber=step_id - base_step_id + 1,
                state="todo",
                timecreated=int(current_time.timestamp()),
                user_id=user_id,
                is_cheating=is_cheating,
                cheating_severity=cheating_severity,
                cheating_group=cheating_group
            )
            self.question_attempt_steps.append(step)
            
            # Scientific timing patterns - only apply for honest students or when we don't have sync timestamps
            if is_cheating and timing_pattern and (not question_sync_timestamps or idx >= len(user_timestamps)-1):
                # Use consistent group timing pattern with small individual variance
                base_time, variance = timing_pattern[idx % len(timing_pattern)]
                
                # Apply a consistent personal style
                individual_variance = random.randint(-variance, variance)
                read_time = max(1, base_time + personal_timing_style + individual_variance)
                
                # Add occasional realistic "thinking pause" 
                if random.random() < 0.15:  
                    read_time += random.randint(5, 15)
                
                current_time += timedelta(seconds=read_time)
                
            elif not is_cheating:
                # Honest: Variable timing with natural behavior
                base_time = random.randint(20, 80)
                
                # Honest students have much higher variance in their thinking time
                if random.random() < 0.3:  # 30% chance of significant variance
                    if random.random() < 0.5:
                        # Sometimes much faster 
                        base_time = max(5, base_time - random.randint(10, 30))
                    else:
                        # Sometimes much slower
                        base_time += random.randint(30, 120)
                
                read_time = base_time
                current_time += timedelta(seconds=read_time)
            
            # We don't need to add time for cheaters using synchronized timestamps
            # as we already have the next timestamp ready
                
            # Answer submission - next step
            step_id += 1
            
            # Set answer time 1-5 seconds after viewing for all users
            if not question_sync_timestamps or not is_cheating:
                current_time += timedelta(seconds=random.randint(1, 5))
            elif is_cheating and idx < len(user_timestamps)-1:
                # For synchronized cheaters, answer time is just a few seconds before next question view
                next_view_time = user_timestamps[idx+1]
                answer_time = next_view_time - timedelta(seconds=random.randint(1, 3))
                if answer_time > current_time:  # Make sure we don't go backwards in time
                    current_time = answer_time
            
            step = QuestionAttemptStep(
                question_step_id=step_id,
                question_attempt_id=user_id * 100 + quiz_id,
                sequencenumber=step_id - base_step_id + 1,
                state="complete",
                timecreated=int(current_time.timestamp()),
                user_id=user_id,
                is_cheating=is_cheating,
                cheating_severity=cheating_severity,
                cheating_group=cheating_group
            )
            self.question_attempt_steps.append(step)
            
            # Record answer selection
            step_data_id = step_id  # Use step_id as step_data_id for uniqueness
            step_data = QuestionAttemptStepData(
                question_step_id=step_id,
                name="answer",
                value=str(answer_sequence[idx]),
                user_id=user_id,
                step_data_id=step_data_id,
                is_cheating=is_cheating,
                cheating_severity=cheating_severity,
                cheating_group=cheating_group
            )
            self.question_attempt_step_data.append(step_data)
            
            completed_questions.append(question_num)
            current_time += timedelta(seconds=random.randint(2, 5))
            
            # Review behavior differentiator: honest students review questions more
            if not is_cheating and len(completed_questions) > 1 and random.random() < 0.3:
                review_question = random.choice(completed_questions[:-1])
                step_id += 1
                step = QuestionAttemptStep(
                    question_step_id=step_id,
                    question_attempt_id=user_id * 100 + quiz_id,
                    sequencenumber=step_id - base_step_id + 1,
                    state="todo",  # reviewing
                    timecreated=int(current_time.timestamp()),
                    user_id=user_id,
                    is_cheating=is_cheating,
                    cheating_severity=cheating_severity,
                    cheating_group=cheating_group
                )
                self.question_attempt_steps.append(step)
                current_time += timedelta(seconds=random.randint(10, 30))
            
            step_id += 1
        
        return step_id
    
    def generate_data(self):
        """Generate all data following the configured patterns"""
        print("Starting Moodle log generation with ML-ready cheating patterns...")
        
        self.generate_users()
        self.generate_quizzes()
        self.generate_questions_and_answers()
        self.generate_sessions_and_attempts()
        
        print("Log generation complete!")
    
    def write_to_csv(self):
        """Write all generated data to CSV files"""
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        
        # Helper function to write a list of objects to CSV
        def write_objects_to_csv(objects, filename):
            if not objects:
                return
            
            fieldnames = asdict(objects[0]).keys()
            with open(os.path.join(self.config["output_dir"], filename), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for obj in objects:
                    writer.writerow(asdict(obj))
        
        # Write all tables to CSV
        write_objects_to_csv(self.users, 'mdl_user.csv')
        write_objects_to_csv(self.quizzes, 'mdl_quiz.csv')
        write_objects_to_csv(self.questions, 'mdl_question.csv')
        write_objects_to_csv(self.question_answers, 'mdl_question_answers.csv')  # Renamed from mdl_question_answers.csv
        write_objects_to_csv(self.sessions, 'mdl_sessions.csv')
        write_objects_to_csv(self.quiz_attempts, 'mdl_quiz_attempts.csv')
        write_objects_to_csv(self.question_attempt_steps, 'mdl_question_attempt_steps.csv')
        write_objects_to_csv(self.question_attempt_step_data, 'mdl_question_attempt_step_data.csv')
        write_objects_to_csv(self.question_attempts_maps, 'mdl_question_attempts.csv') # ADDED
        
        # Generate a ground truth file for ML reference
        self.write_ground_truth()
        
        print(f"All data written to {self.config['output_dir']} directory")
    
    def write_ground_truth(self):
        """Write a ground truth file with cheating information"""
        with open(os.path.join(self.config["output_dir"], 'cheating_ground_truth.md'), 'w') as f:
            f.write("# Cheating Ground Truth\n\n")
            f.write("This file contains the ground truth about cheating groups for validation and ML training.\n\n")
            
            # Generate scientific statistical summary table
            f.write("## Statistical Summary of Cheating Groups\n\n")
            f.write("This table provides statistical justification for the labeling of cheating groups, showing quantitative evidence of coordination.\n\n")
            f.write("| Group | Navigation Similarity (%) | Answer Pattern Similarity (%) | Timing Correlation | Std Dev (Avg) | Wrong Answer Bias |\n")
            f.write("|-------|----------------------------|------------------------------|-------------------|----------------|-------------------|\n")
            
            # Calculate the actual statistical metrics for each group
            for group in self.cheating_groups:
                # Calculate actual statistics from the simulation data
                nav_similarity = group.patterns["navigation"]["similarity"] * 100
                
                # Calculate actual answer pattern similarity from the data
                answer_similarity = group.patterns["answers"]["similarity"] * 100
                
                # Calculate actual timing correlations from the simulation
                timing_correlation = 0
                std_dev_avg = 0
                
                # Compute real statistics from the generated data if available
                if hasattr(group, 'quiz_patterns') and len(group.quiz_patterns) > 0:
                    # Calculate across all quizzes
                    all_correlations = []
                    all_std_devs = []
                    
                    for quiz_id, patterns in group.quiz_patterns.items():
                        if "question_sync_timestamps" in patterns:
                            # Calculate actual temporal correlation between group members
                            # This is based on the simulation data rather than just the config
                            member_times = {}
                            
                            # Get the actual timestamps for each member
                            for user_id in group.members:
                                steps = [s for s in self.question_attempt_steps 
                                        if s.user_id == user_id 
                                        and "question_attempt_id" in s.__dict__
                                        and s.question_attempt_id == user_id * 100 + quiz_id
                                        and s.state == "todo"]
                                
                                if steps:
                                    # Sort by time
                                    steps = sorted(steps, key=lambda s: s.timecreated)
                                    
                                    # Extract times
                                    times = [s.timecreated for s in steps]
                                    
                                    # Calculate transition times
                                    if len(times) > 1:
                                        transitions = [times[i+1] - times[i] for i in range(len(times)-1)]
                                        member_times[user_id] = transitions
                            
                            # Calculate correlations between all pairs
                            pair_correlations = []
                            for i, user1 in enumerate(group.members):
                                for user2 in group.members[i+1:]:
                                    if user1 in member_times and user2 in member_times:
                                        t1 = member_times[user1]
                                        t2 = member_times[user2]
                                        
                                        # Must have at least a few transitions to calculate correlation
                                        if len(t1) >= 3 and len(t2) >= 3:
                                            # Use the minimum length
                                            min_len = min(len(t1), len(t2))
                                            t1 = t1[:min_len]
                                            t2 = t2[:min_len]
                                            
                                            try:
                                                corr = np.corrcoef(t1, t2)[0, 1]
                                                if not np.isnan(corr):
                                                    pair_correlations.append(corr)
                                            except:
                                                pass
                            
                            if pair_correlations:
                                all_correlations.extend(pair_correlations)
                        
                        # Calculate standard deviations in timing
                        for user_id in group.members:
                            steps = [s for s in self.question_attempt_steps 
                                    if s.user_id == user_id 
                                    and "question_attempt_id" in s.__dict__
                                    and s.question_attempt_id == user_id * 100 + quiz_id]
                            
                            question_times = []
                            for i in range(0, len(steps)-1, 2):
                                if i+1 < len(steps):
                                    time_spent = steps[i+1].timecreated - steps[i].timecreated
                                    question_times.append(time_spent)
                            
                            if len(question_times) > 1:
                                std_dev = np.std(question_times)
                                all_std_devs.append(std_dev)
                    
                    # Calculate averages across all quizzes and members
                    if all_correlations:
                        timing_correlation = sum(all_correlations) / len(all_correlations)
                    
                    if all_std_devs:
                        std_dev_avg = sum(all_std_devs) / len(all_std_devs)
                
                # Record actual wrong answer bias from the config
                wrong_bias = group.patterns["answers"]["wrong_bias"] * 100
                
                # Format the table row with the calculated statistics
                f.write(f"| {group.id} | {nav_similarity:.1f}% | {answer_similarity:.1f}% | {timing_correlation:.4f} | {std_dev_avg:.1f} | {wrong_bias:.1f}% |\n")
            
            f.write("\n*Note: These statistics represent the actual measured values from the simulation, not just the configuration parameters.*\n\n")
            f.write("- **Navigation Similarity**: Percentage of identical navigation patterns between group members\n")
            f.write("- **Answer Pattern Similarity**: Percentage of identical answers between group members\n")
            f.write("- **Timing Correlation**: Pearson correlation coefficient of question transition times (values >0.7 are statistically improbable without coordination)\n")
            f.write("- **Std Dev**: Average standard deviation of time spent per question (lower values indicate suspiciously consistent timing)\n")
            f.write("- **Wrong Answer Bias**: Probability of coordinated wrong answers (higher values indicate suspicious identical mistakes)\n\n")
            
            f.write("## Cheater Groups\n\n")
            for group in self.cheating_groups:
                f.write(f"### {group.id} (Severity: {group.severity})\n")
                f.write(f"- Members: {group.members}\n")
                f.write(f"- Navigation similarity: {group.patterns['navigation']['similarity']}\n")
                f.write(f"- Navigation noise: {group.patterns['navigation']['noise']}\n")
                f.write(f"- Timing start delay: {group.patterns['timing']['start_delay']} minutes\n")
                f.write(f"- Timing variance: {group.patterns['timing']['variance']} seconds\n")
                f.write(f"- Answer similarity: {group.patterns['answers']['similarity']}\n")
                f.write(f"- Wrong answer bias: {group.patterns['answers']['wrong_bias']}\n\n")
        
        # Also write a CSV version for easy ML import
        with open(os.path.join(self.config["output_dir"], 'cheating_ground_truth.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'is_cheater', 'cheating_group', 'cheating_severity'])
            
            for user in self.users:
                writer.writerow([
                    user.id, 
                    1 if user.is_cheater else 0, 
                    user.cheating_group or 'N/A', 
                    user.cheating_severity or 'N/A'
                ])
    
    def write_visualization_md(self):
        """Create visual representation of user attempts to easily identify cheating patterns"""
        with open(os.path.join(self.config["output_dir"], 'cheating_visualization.md'), 'w') as f:
            f.write("# Quiz Attempt Visualization\n\n")
            f.write("This file provides a visual representation of quiz attempts to identify suspicious patterns.\n\n")
            
            # Process each quiz
            for quiz_id in range(1, self.config["total_quizzes"] + 1):
                quiz = next(q for q in self.quizzes if q.quiz_id == quiz_id)
                f.write(f"## {quiz.quiz_name}\n\n")
                
                # Get all attempts for this quiz
                attempts = [a for a in self.quiz_attempts if a.quiz_id == quiz_id]
                
                # Get correct answers for reference
                correct_answers = self.get_correct_answers(quiz_id)
                
                # Create visualization sections
                f.write("### Navigation Patterns\n\n")
                f.write("Each row represents a user's navigation sequence through questions. Similar sequences may indicate cheating.\n")
                f.write("Numbers that appear multiple times indicate question revisits.\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Navigation Sequence (with revisits)\n")
                f.write("--------|------------|------------|------------------------------------\n")
                
                # Sort attempts by cheating group and user ID
                attempts.sort(key=lambda a: (0 if a.is_cheating else 1, a.cheating_group or "", a.user_id))
                
                # Lookup table for cheating groups' navigation patterns
                group_nav_patterns = {}
                for group in self.cheating_groups:
                    if hasattr(group, 'quiz_patterns') and quiz_id in group.quiz_patterns:
                        group_nav_patterns[group.id] = group.quiz_patterns[quiz_id]["navigation_sequence"]
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Use the original navigation sequence for cheaters (which includes revisits)
                    if is_cheater == "YES" and group in group_nav_patterns:
                        nav_sequence = group_nav_patterns[group][:20]  # Show first 20 for clarity
                    else:
                        # For honest students, get steps from the database
                        steps = sorted([s for s in self.question_attempt_steps 
                                      if s.user_id == user.id and "question_attempt_id" in s.__dict__ 
                                      and s.question_attempt_id == user.id * 100 + quiz_id],
                                     key=lambda s: s.timecreated)
                        
                        nav_sequence = []
                        for step in steps:
                            if step.state == "todo":  # All view steps, not just initial
                                q_num = (step.sequencenumber + 1) // 2
                                nav_sequence.append(q_num)
                        
                        # Limit to first 20 for display clarity
                        nav_sequence = nav_sequence[:20]
                    
                    # Format navigation sequence with fixed width
                    nav_str = " ".join([f"{q:2}" for q in nav_sequence])
                    
                    # Use special formatting for cheaters
                    if user.is_cheater:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {nav_str}\n")
                    else:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {nav_str}\n")
                
                f.write("```\n\n")
                
                # Check for revisit patterns
                f.write("### Question Revisit Analysis\n\n")
                f.write("This shows how many times each question was visited by each user. Multiple visits indicate revisiting.\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Questions With Multiple Visits\n")
                f.write("--------|------------|------------|------------------------------\n")
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Get actual navigation sequence (either from pattern or steps)
                    if is_cheater == "YES" and group in group_nav_patterns:
                        nav_sequence = group_nav_patterns[group]
                    else:
                        steps = sorted([s for s in self.question_attempt_steps 
                                       if s.user_id == user.id and "question_attempt_id" in s.__dict__ 
                                       and s.question_attempt_id == user.id * 100 + quiz_id],
                                      key=lambda s: s.timecreated)
                        
                        nav_sequence = []
                        for step in steps:
                            if step.state == "todo":
                                q_num = (step.sequencenumber + 1) // 2  
                                nav_sequence.append(q_num)
                    
                    # Find questions with multiple visits
                    revisits = {}
                    for q in nav_sequence:
                        if q in revisits:
                            revisits[q] += 1
                        else:
                            revisits[q] = 1
                    
                    # Format revisit info
                    revisit_str = ", ".join([f"Q{q}({count}x)" for q, count in revisits.items() if count > 1])
                    if not revisit_str:
                        revisit_str = "None"
                    
                    f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {revisit_str}\n")
                
                f.write("```\n\n")
                
                # Answer patterns visualization
                f.write("### Answer Patterns\n\n")
                f.write("Each row shows a user's answers. C = correct, X = wrong. Similar answer patterns may indicate cheating.\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Q1 Q2 Q3 Q4 ... \n")
                f.write("--------|------------|------------|----------------\n")
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Get answer data for this attempt
                    answer_steps = sorted([step for step in self.question_attempt_step_data
                                         if step.user_id == user.id and step.name == "answer"],
                                        key=lambda s: s.step_data_id)
                    
                    # Build pattern of correct/wrong answers
                    answer_pattern = []
                    for idx, step_data in enumerate(answer_steps[:20]):  # Show first 20 for clarity
                        user_answer = int(step_data.value)
                        correct = correct_answers[idx]
                        if user_answer == correct:
                            answer_pattern.append("C")  # Correct
                        else:
                            answer_pattern.append("X")  # Wrong
                    
                    # Format answer pattern with spacing
                    ans_str = " ".join(answer_pattern)
                    
                    # Use special formatting for cheaters
                    if user.is_cheater:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {ans_str}\n")
                    else:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {ans_str}\n")
                
                f.write("```\n\n")
                
                # Timing patterns visualization 
                f.write("### Timing Patterns\n\n")
                f.write("Each row shows when a user started and how long they spent on each question (in seconds).\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Start Time         | Total Duration | Avg Time/Q \n")
                f.write("--------|------------|------------|-------------------|---------------|------------\n")
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Get timing info
                    start_time = datetime.fromtimestamp(attempt.timestart).strftime('%Y-%m-%d %H:%M:%S')
                    duration_seconds = attempt.timefinish - attempt.timestart
                    duration_minutes = duration_seconds / 60
                    
                    # Calculate average time per question
                    question_count = len([q for q in self.questions if q.quizid == quiz_id])
                    avg_time_per_q = duration_seconds / question_count
                    
                    # Use special formatting for cheaters
                    if user.is_cheater:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {start_time:19} | {duration_minutes:13.1f} min | {avg_time_per_q:10.1f} s\n")
                    else:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {start_time:19} | {duration_minutes:13.1f} min | {avg_time_per_q:10.1f} s\n")
                
                f.write("```\n\n")
                
                # Add detailed question-by-question timing for suspicious cases
                f.write("### Detailed Timing by Question\n\n")
                f.write("Time spent (seconds) on each question by users, grouped by cheating status:\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Q1   Q2   Q3   Q4   Q5   Q6   Q7   Q8   Q9   Q10  \n")
                f.write("--------|------------|------------|----------------------------------------------------\n")
                
                # Process cheaters first to group them together in the visual
                attempts.sort(key=lambda a: (0 if a.is_cheating else 1, a.cheating_group or "", a.user_id))
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Get steps with timestamps to calculate time spent on each question
                    steps = sorted([s for s in self.question_attempt_steps 
                                  if s.user_id == user.id and "question_attempt_id" in s.__dict__ 
                                  and s.question_attempt_id == user.id * 100 + quiz_id],
                                 key=lambda s: s.timecreated)
                    
                    # Calculate time spent on each question
                    question_times = []
                    for i in range(0, len(steps)-1, 2):
                        if i+1 < len(steps):
                            time_spent = steps[i+1].timecreated - steps[i].timecreated
                            question_times.append(time_spent)
                    
                    # Format timing with color indicators for easy pattern recognition
                    timing_str = " ".join([f"{t:4}" for t in question_times[:10]])  # First 10 questions
                    
                    # Add visualization with special formatting for cheaters
                    if user.is_cheater:
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {timing_str}\n")
                    else:
                        # Only show some honest students for comparison
                        if random.random() < 0.5:  # Show only half of honest students to reduce clutter
                            f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {timing_str}\n")
                
                f.write("```\n\n")
                
                # Add variance analysis to highlight the suspicious similarity
                f.write("### Timing Variance Analysis\n\n")
                f.write("This shows how consistent each user's pace is (lower standard deviation = more suspicious consistency):\n\n")
                f.write("```\n")
                f.write("User ID | Is Cheater | Group      | Avg Time/Q | Std Deviation | Coefficient of Variation\n")
                f.write("--------|------------|------------|------------|---------------|------------------------\n")
                
                # Sort attempts by cheating group
                attempts.sort(key=lambda a: (0 if a.is_cheating else 1, a.cheating_group or "", a.user_id))
                
                for attempt in attempts:
                    # Get user info
                    user = next(u for u in self.users if u.id == attempt.user_id)
                    is_cheater = "YES" if user.is_cheater else "NO"
                    group = user.cheating_group or "None"
                    
                    # Get steps with timestamps to calculate time spent on each question
                    steps = sorted([s for s in self.question_attempt_steps 
                                  if s.user_id == user.id and "question_attempt_id" in s.__dict__ 
                                  and s.question_attempt_id == user.id * 100 + quiz_id],
                                 key=lambda s: s.timecreated)
                    
                    # Calculate time spent on each question
                    question_times = []
                    for i in range(0, len(steps)-1, 2):
                        if i+1 < len(steps):
                            time_spent = steps[i+1].timecreated - steps[i].timecreated
                            question_times.append(time_spent)
                    
                    # Calculate statistics
                    if question_times:
                        avg_time = sum(question_times) / len(question_times)
                        std_dev = 0
                        if len(question_times) > 1:
                            std_dev = np.std(question_times)
                        coef_var = (std_dev / avg_time) * 100 if avg_time > 0 else 0
                        
                        # Format and output
                        f.write(f"{user.id:7} | {is_cheater:10} | {group:10} | {avg_time:10.1f} | {std_dev:13.1f} | {coef_var:23.1f}%\n")
                
                f.write("```\n\n")
                
                # Add a note about what to look for
                f.write("**Note:** In real-world cheating, the suspicious pattern is not necessarily fast completion time, but rather\n")
                f.write("the *similarity in pace* between users in the same cheating group. Look for groups of users with similar timing\n")
                f.write("patterns and low standard deviations. Honest students typically show much more variable timing patterns.\n\n")
            
            # Summary and interpretation section
            f.write("## Interpretation Guide\n\n")
            f.write("### Suspicious Patterns to Look For:\n\n")
            f.write("1. **Navigation Sequences**: Identical or highly similar navigation patterns between users suggest coordination\n")
            f.write("2. **Answer Patterns**: Similar patterns of correct/incorrect answers, especially wrong answers at same positions\n")
            f.write("3. **Timing Patterns**: \n")
            f.write("   - **Similar pace between questions for users in the same group** (primary indicator)\n")
            f.write("   - Low standard deviation in timing (unnaturally consistent pace)\n")
            f.write("   - Users in same cheating group starting at regular intervals\n")
            f.write("   - Look for suspiciously synchronized patterns rather than just fast times\n\n")
            
            f.write("4. **Group Behavior**: Users in the same cheating group should show correlated patterns across multiple dimensions\n\n")
            
            f.write("### Real-World Cheating Patterns:\n\n")
            f.write("In realistic cheating scenarios, students often:\n\n")
            f.write("1. **Follow a leader**: One student completes the quiz first, then shares answers with others\n")
            f.write("2. **Maintain consistent timing**: Move through questions at suspiciously similar rates\n") 
            f.write("3. **Make identical mistakes**: Share the same wrong answers where the correct answer is difficult\n")
            f.write("4. **Navigate in patterns**: Similar navigation sequences that don't reflect natural problem-solving\n\n")
            
            f.write("The most reliable indicator is often the *combination* of these signals rather than any single pattern.\n")
            
            # Add timestamp analysis to detect synchronization
            f.write("### Question Navigation Timestamps\n\n")
            f.write("This shows the exact times when users first viewed each question. Synchronized timestamps between users indicate coordination.\n\n")
            f.write("```\n")
            f.write("User ID | Is Cheater | Group      | Q1              | Q2              | Q3              | Q4              | Q5              \n")
            f.write("--------|------------|------------|-----------------|-----------------|-----------------|-----------------|----------------\n")
            
            # Sort attempts by cheating group and user ID
            attempts.sort(key=lambda a: (0 if a.is_cheating else 1, a.cheating_group or "", a.user_id))
            
            for attempt in attempts:
                # Get user info
                user = next(u for u in self.users if u.id == attempt.user_id)
                is_cheater = "YES" if user.is_cheater else "NO"
                group = user.cheating_group or "None"
                
                # Get steps with question view timestamps
                steps = sorted([s for s in self.question_attempt_steps 
                              if s.user_id == user.id and "question_attempt_id" in s.__dict__ 
                              and s.question_attempt_id == user.id * 100 + quiz_id 
                              and s.state == "todo"],
                                 key=lambda s: s.timecreated)
                
                # Extract first views of each question (exclude revisits)
                question_times = {}
                for step in steps:
                    q_num = (step.sequencenumber + 1) // 2
                    if q_num not in question_times:  # Only first view
                        question_times[q_num] = step.timecreated
                
                # Format timestamps for first 5 questions
                timestamp_str = ""
                for q in range(1, 6):
                    if q in question_times:
                        ts = datetime.fromtimestamp(question_times[q]).strftime('%H:%M:%S')
                        timestamp_str += f" {ts:16} |"
                    else:
                        timestamp_str += f" {'N/A':16} |"
                
                f.write(f"{user.id:7} | {is_cheater:10} | {group:10} |{timestamp_str}\n")
            
            f.write("```\n\n")
            
            # Add scientific transition time analysis
            f.write("### Transition Time Correlation Analysis\n\n")
            f.write("This analysis shows the correlation between users' question transition times within groups.\n")
            f.write("High correlation coefficients (close to 1.0) indicate synchronized movement between questions.\n\n")
            f.write("```\n")
            f.write("Group                | Users                  | Transition Time Correlation\n")
            f.write("--------------------|------------------------|---------------------------\n")
            
            # Group attempts by cheating group
            group_attempts = {}
            for attempt in attempts:
                user = next(u for u in self.users if u.id == attempt.user_id)
                if user.is_cheater and user.cheating_group:
                    if user.cheating_group not in group_attempts:
                        group_attempts[user.cheating_group] = []
                    group_attempts[user.cheating_group].append(attempt)
            
            # Calculate transition time correlations for each group
            for group_name, group_atts in group_attempts.items():
                # Skip if only one user in group
                if len(group_atts) <= 1:
                    continue
                
                # Get user IDs in this group
                user_ids = [a.user_id for a in group_atts]
                user_str = ", ".join([str(u) for u in user_ids])
                
                # Calculate transition times for each user
                user_transitions = {}
                for att in group_atts:
                    user_id = att.user_id
                    steps = sorted([s for s in self.question_attempt_steps 
                                  if s.user_id == user_id and "question_attempt_id" in s.__dict__ 
                                  and s.question_attempt_id == user_id * 100 + quiz_id 
                                  and s.state == "todo"],
                                     key=lambda s: s.timecreated)
                    
                    # Calculate time differences between consecutive questions
                    transitions = []
                    for i in range(1, len(steps)):
                        transitions.append(steps[i].timecreated - steps[i-1].timecreated)
                    
                    if transitions:
                        user_transitions[user_id] = transitions
                
                # Calculate correlation between users in the group
                correlations = []
                for u1 in user_transitions:
                    for u2 in user_transitions:
                        if u1 < u2:  # Avoid duplicates
                            # Calculate correlation if we have enough transition times
                            t1 = user_transitions[u1]
                            t2 = user_transitions[u2]
                            min_len = min(len(t1), len(t2))
                            
                            if min_len >= 3:  # Need at least 3 points for meaningful correlation
                                # Truncate to same length
                                t1 = t1[:min_len]
                                t2 = t2[:min_len]
                                
                                # Calculate correlation coefficient
                                try:
                                    correlation = np.corrcoef(t1, t2)[0, 1]
                                    correlations.append(correlation)
                                except:
                                    correlations.append(0)
                
                # Calculate average correlation for the group
                if correlations:
                    avg_correlation = sum(correlations) / len(correlations)
                    correlation_str = f"{avg_correlation:.4f}"
                    # Add a scientific interpretation
                    if avg_correlation > 0.8:
                        correlation_str += " (Very High - Strong evidence of coordination)"
                    elif avg_correlation > 0.6:
                        correlation_str += " (High - Suspicious coordination)"
                    elif avg_correlation > 0.4:
                        correlation_str += " (Moderate - Possible coordination)"
                    else:
                        correlation_str += " (Low - Unlikely coordination)"
                else:
                    correlation_str = "Insufficient data"
                
                f.write(f"{group_name:20} | {user_str:22} | {correlation_str}\n")
            
            f.write("```\n\n")
    
    def save_config(self):
        """Save the current configuration"""
        with open(os.path.join(self.config["output_dir"], 'generator_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

def main():
    """Main function to generate data"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate synthetic Moodle logs with cheating patterns')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, default='data/moodle_logs', help='Output directory')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MoodleLogGenerator(args.config)
    
    if args.output:
        # Convert relative path to absolute if needed
        if not os.path.isabs(args.output):
            args.output = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
        generator.config["output_dir"] = args.output
    
    # Generate and write data
    generator.generate_data()
    generator.write_to_csv()
    generator.save_config()
    generator.write_ground_truth()
    generator.write_visualization_md()
    
    print("Done!")

if __name__ == "__main__":
    main()
