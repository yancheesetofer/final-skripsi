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
from collections import defaultdict

warnings.filterwarnings('ignore')
fake = Faker()

DEFAULT_CONFIG = {
  "total_users": 45,
  "total_quizzes": 4,
  "questions_per_quiz": 20, # Reduced for more manageable visualization output
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
    "low_severity": { # Added a low severity group for more diverse data
      "count": 1,
      "size": 5,
      "patterns": {
        "navigation": {
          "similarity": 0.55,
          "noise": 0.45
        },
        "timing": {
          "start_delay": 15,
          "variance": 30,
          "completion_speed": "slow" # Changed to slow for variety
        },
        "answers": {
          "similarity": 0.55,
          "wrong_bias": 0.4
        }
      }
    }
  },
  "output_format": "csv",
  "output_dir": "data/moodle_logs_refined", # More generic default
  "seed": 12345
}

@dataclass
class CheatingGroupConfig:
    id: str
    severity: str
    members: List[int]
    patterns: Dict[str, Any]
    quiz_patterns: Dict[int, Dict[str, Any]] = field(default_factory=dict)


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
    cheating_group_id: Optional[str] = None
    cheating_severity: Optional[str] = None

@dataclass
class Quiz:
    quiz_id: int
    course: int
    quiz_name: str
    timeopen: int
    timeclose: int
    timelimit: int

@dataclass
class Question:
    id: int # Internal ID for question
    qtype: str
    name: str
    questiontext: str
    defaultmark: float
    penalty: float
    quizid: int # Links to Quiz.quiz_id

@dataclass
class QuestionAnswer:
    question_answers_id: int
    questionid: int # Links to Question.id
    answer_text: str
    fraction: float

@dataclass
class Session:
    session_id: int
    user_id: int
    timecreated: int
    lastip: str
    sessdata: str = ""

@dataclass
class QuizAttempt:
    attempt_id: int
    quiz_id: int
    user_id: int
    question_usage_id: int
    timestart: int
    timefinish: int
    state: str
    sumgrades: float

@dataclass
class QuestionUsage:
    question_usage_id: int
    context_id: int

@dataclass
class QuestionAttemptReal: # mdl_question_attempts
    question_attempt_id: int # PK
    question_usage_id: int   # FK to QuestionUsage
    questionid: int          # FK to Question.id (actual question displayed)
    maxmark: float

@dataclass
class QuestionAttemptStep:
    question_step_id: int # PK
    question_attempt_id: int # FK to QuestionAttemptReal
    sequencenumber: int
    state: str
    timecreated: int

@dataclass
class QuestionAttemptStepData:
    step_data_id: int # PK
    question_step_id: int # FK to QuestionAttemptStep
    name: str
    value: str

@dataclass
class QuizGrade:
    quiz_grades_id: int # PK
    quiz_id: int
    user_id: int
    final_grade: float


class MoodleLogGenerator:
    def __init__(self, config_file=None):
        self.config = DEFAULT_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        
        self.users_internal: List[User] = []
        self.quizzes: List[Quiz] = []
        self.questions_internal: List[Question] = []
        self.question_answers: List[QuestionAnswer] = []
        self.sessions: List[Session] = []
        self.quiz_attempts: List[QuizAttempt] = []
        self.question_usages: List[QuestionUsage] = []
        self.question_attempts_real: List[QuestionAttemptReal] = []
        self.question_attempt_steps: List[QuestionAttemptStep] = []
        self.question_attempt_step_data: List[QuestionAttemptStepData] = []
        self.quiz_grades: List[QuizGrade] = []
        
        self.next_id = {
            "user": 1, "quiz": 1, "question_internal": 1, "Youtube": 1,
            "session": 1, "quiz_attempt": 1, "question_usage": 1, 
            "question_attempt_real": 1, "question_attempt_step": 1,
            "step_data": 1, "quiz_grade": 1
        }
        
        self.cheating_groups_config: List[CheatingGroupConfig] = []
        self.cheater_user_ids: List[int] = []
        self.setup_cheating_groups()
        
        self.base_date = datetime.strptime(self.config["base_date"], "%Y-%m-%d %H:%M:%S")
    
    def setup_cheating_groups(self):
        user_id_counter = 1
        total_users_in_config = self.config["total_users"]
        users_assigned_to_groups = 0

        for severity, config_details in self.config["cheating_groups"].items():
            for i in range(config_details["count"]):
                if users_assigned_to_groups + config_details["size"] > total_users_in_config:
                    print(f"Warning: Not enough users to create all cheating groups. Skipping group {severity}_{i+1}")
                    continue

                group_name = f"group_{severity}_{i+1}"
                members = list(range(user_id_counter, user_id_counter + config_details["size"]))
                user_id_counter += config_details["size"]
                users_assigned_to_groups += config_details["size"]
                
                group = CheatingGroupConfig(
                    id=group_name,
                    severity=severity,
                    members=members,
                    patterns=config_details["patterns"]
                )
                self.cheating_groups_config.append(group)
                self.cheater_user_ids.extend(members)
        
        if len(self.cheater_user_ids) > total_users_in_config: # Should be caught by the loop above too
             raise ValueError("Assigned more cheaters than total users. Adjust config.")
        print(f"Created {len(self.cheating_groups_config)} cheating groups with {len(self.cheater_user_ids)} total cheaters.")
        print(f"Remaining {total_users_in_config - len(self.cheater_user_ids)} users will be honest.")


    def generate_users(self):
        for user_id_val in range(1, self.config["total_users"] + 1):
            is_cheater_flag = user_id_val in self.cheater_user_ids
            cheating_group_identifier = None
            cheating_sev = None
            
            if is_cheater_flag:
                for group_conf in self.cheating_groups_config:
                    if user_id_val in group_conf.members:
                        cheating_group_identifier = group_conf.id
                        cheating_sev = group_conf.severity
                        break
            
            user_obj = User(
                id=user_id_val, username=fake.user_name(), password=fake.sha256(),
                firstname=fake.first_name(), lastname=fake.last_name(), email=fake.email(),
                lastaccess=int((self.base_date + timedelta(days=random.randint(0,30))).timestamp()), # More varied lastaccess
                is_cheater=is_cheater_flag, cheating_group_id=cheating_group_identifier, cheating_severity=cheating_sev
            )
            self.users_internal.append(user_obj)
        print(f"Generated {len(self.users_internal)} internal user representations")

    def generate_quiz_topics(self):
        topics = [
            "Intro to Python", "Advanced Algorithms", "DB Management",
            "Network Security", "OS Concepts", "Software Design Patterns",
            "AI Ethics", "ML Foundations", "3D Rendering", "Full-Stack Web Dev"
        ] # Slightly modified
        return random.sample(topics, min(len(topics), self.config["total_quizzes"]))

    def generate_quizzes(self):
        topics = self.generate_quiz_topics()
        for i in range(1, self.config["total_quizzes"] + 1):
            topic = topics[i-1] if i <= len(topics) else f"Quiz Topic {i}"
            timeopen_dt = self.base_date + timedelta(days=(i-1)*7 + random.randint(0,2)) # Stagger quiz starts slightly
            timeclose_dt = timeopen_dt + timedelta(days=random.randint(2,5)) # Vary closing time
            
            quiz_obj = Quiz(
                quiz_id=self.next_id["quiz"], course=random.randint(101, 105), 
                quiz_name=f"Quiz {i}: {topic}",
                timeopen=int(timeopen_dt.timestamp()),
                timeclose=int(timeclose_dt.timestamp()),
                timelimit=random.choice([1800, 2700, 3600, 5400, 7200]) 
            )
            self.quizzes.append(quiz_obj)
            self.next_id["quiz"] += 1
        print(f"Generated {len(self.quizzes)} quizzes")

    def generate_questions_and_answers(self):
        for quiz_obj in self.quizzes:
            for q_num in range(1, self.config["questions_per_quiz"] + 1):
                q_id = self.next_id["question_internal"]
                question_obj = Question(
                    id=q_id, qtype="multichoice", name=f"Q{q_num}",
                    questiontext=f"Content of Question {q_num} for {quiz_obj.quiz_name}", # More descriptive
                    defaultmark=round(random.uniform(0.5, 2.0),1), # Varied marks
                    penalty=round(random.uniform(0.0, 0.33), 2), quizid=quiz_obj.quiz_id
                )
                self.questions_internal.append(question_obj)
                self.next_id["question_internal"] += 1
                
                options = [f"Option {chr(65+opt_idx)} for Q{q_num}" for opt_idx in range(4)] # A, B, C, D
                random.shuffle(options)
                correct_answer_idx = random.randint(0, 3)

                for ans_idx in range(4):
                    ans_id = self.next_id["Youtube"]
                    answer_obj = QuestionAnswer(
                        question_answers_id=ans_id, questionid=q_id,
                        answer_text=options[ans_idx],
                        fraction=1.0 if ans_idx == correct_answer_idx else ( -0.33 if random.random() < 0.1 else 0.0) # Occasional negative fraction for distractor
                    )
                    self.question_answers.append(answer_obj)
                    self.next_id["Youtube"] += 1
        print(f"Generated {len(self.questions_internal)} internal questions and {len(self.question_answers)} answers")

    def get_correct_answers_for_quiz(self, quiz_id_val: int) -> Dict[int, int]:
        correct_answers_map = {} # Maps question_id to correct answer_text index
        questions_for_this_quiz = [q for q in self.questions_internal if q.quizid == quiz_id_val]
        for question_intern in questions_for_this_quiz:
            answers_for_this_q = sorted([a for a in self.question_answers if a.questionid == question_intern.id], key=lambda x: x.question_answers_id) # Ensure consistent order
            try:
                correct_choice_idx = next((i for i, a in enumerate(answers_for_this_q) if a.fraction >= 1.0), 0) # Allow fraction >= 1
                correct_answers_map[question_intern.id] = correct_choice_idx
            except StopIteration:
                 correct_answers_map[question_intern.id] = 0 # Default if no correct answer found (should not happen)
                 print(f"Warning: No correct answer found for question ID {question_intern.id} in quiz {quiz_id_val}. Defaulting to option 0.")
        return correct_answers_map

    def generate_navigation_sequence(self, question_ids_in_quiz: List[int], is_cheater=False, pattern_config=None, base_sequence: Optional[List[int]]=None):
        num_questions = len(question_ids_in_quiz)
        if not question_ids_in_quiz: return []

        if is_cheater and base_sequence and pattern_config:
            sequence = base_sequence.copy()
            # Apply noise by swapping some elements based on similarity config
            # Higher similarity means fewer swaps (more adherence to base)
            # Noise in config is how much it *deviates*
            # So, number of swaps is proportional to (1 - similarity) or noise
            # similarity = pattern_config["navigation"]["similarity"]
            noise = pattern_config["navigation"]["noise"]
            swaps = int(num_questions * noise) 

            for _ in range(swaps):
                if num_questions > 1: # Ensure there are at least two elements to swap
                    i, j = random.sample(range(num_questions), 2)
                    sequence[i], sequence[j] = sequence[j], sequence[i]
        else: # Honest user or no base sequence for cheater (should not happen for cheater if leader exists)
            sequence = random.sample(question_ids_in_quiz, len(question_ids_in_quiz)) # Start with a random shuffle
        
        # Add revisits (more for honest users, controlled for cheaters)
        revisit_count_factor = 0.25 if is_cheater else 0.4 # Cheaters might revisit less to avoid suspicion or if copying directly
        revisit_count = random.randint(0, int(num_questions * revisit_count_factor))

        for _ in range(revisit_count):
            if not sequence: continue
            question_to_revisit = random.choice(sequence)
            try:
                first_pos = sequence.index(question_to_revisit)
                # Insert revisit at a random position after its first appearance
                if len(sequence) > first_pos + 1:
                    insert_pos = random.randint(first_pos + 1, len(sequence))
                    sequence.insert(insert_pos, question_to_revisit)
                else: # if it's the last element, insert before it to make it a revisit
                    sequence.insert(len(sequence), question_to_revisit)
            except ValueError: # Should not happen if question_to_revisit is from sequence
                pass
        return sequence


    def generate_answer_selection_sequence(self, nav_sequence: List[int], correct_answers_map: Dict[int, int], 
                                           is_cheater=False, pattern_config=None, base_answer_selections: Optional[List[int]]=None):
        actual_answers_selected = []
        
        if is_cheater and base_answer_selections and pattern_config:
            ans_similarity = pattern_config["answers"]["similarity"]
            wrong_bias = pattern_config["answers"]["wrong_bias"]
            for i, q_id in enumerate(nav_sequence):
                correct_ans_idx = correct_answers_map.get(q_id, 0)
                
                # If copying from leader (base_answer_selections)
                if i < len(base_answer_selections):
                    leader_answer = base_answer_selections[i]
                    if random.random() < ans_similarity: # Follow leader's answer
                        actual_answers_selected.append(leader_answer)
                    # If leader was wrong, and this follower is biased to copy wrong answers
                    elif leader_answer != correct_ans_idx and random.random() < wrong_bias:
                        actual_answers_selected.append(leader_answer) # Copy the wrong answer
                    else: # Deviate: Generate own answer (could be right or wrong)
                        if random.random() < 0.6: # 60% chance of own correct if deviating
                            actual_answers_selected.append(correct_ans_idx)
                        else:
                            wrong_options = [opt_idx for opt_idx in range(4) if opt_idx != correct_ans_idx]
                            actual_answers_selected.append(random.choice(wrong_options) if wrong_options else correct_ans_idx)
                else: # nav_sequence is longer than leader's answers (e.g. more revisits for follower)
                    # Generate own answer for these extra steps
                    if random.random() < 0.5: # Cheaters might still get some right on their own
                        actual_answers_selected.append(correct_ans_idx)
                    else:
                        wrong_options = [opt_idx for opt_idx in range(4) if opt_idx != correct_ans_idx]
                        actual_answers_selected.append(random.choice(wrong_options) if wrong_options else correct_ans_idx)
        else: # Honest User
            for q_id in nav_sequence:
                correct_ans_idx = correct_answers_map.get(q_id, 0)
                # Honest users have a higher chance of getting it right, but not perfect
                if random.random() < 0.75: # 75% chance of correct for honest users
                    actual_answers_selected.append(correct_ans_idx)
                else:
                    wrong_options = [opt_idx for opt_idx in range(4) if opt_idx != correct_ans_idx]
                    actual_answers_selected.append(random.choice(wrong_options) if wrong_options else correct_ans_idx)
        return actual_answers_selected

    def pre_generate_group_quiz_patterns(self):
        for group_conf in self.cheating_groups_config:
            for quiz_obj in self.quizzes:
                question_ids_for_quiz = [q.id for q in self.questions_internal if q.quizid == quiz_obj.quiz_id]
                if not question_ids_for_quiz: continue

                # Leader's navigation sequence (can be somewhat patterned, e.g. sequential for simplicity of base)
                leader_nav_seq = self.generate_navigation_sequence(question_ids_for_quiz, is_cheater=True, pattern_config=group_conf.patterns) # Pass pattern_config
                
                correct_ans_map = self.get_correct_answers_for_quiz(quiz_obj.quiz_id)
                
                # Leader's answers (simulates the "source" answers for the group)
                leader_answers = self.generate_answer_selection_sequence(leader_nav_seq, correct_ans_map, is_cheater=True, pattern_config=group_conf.patterns) # Pass pattern_config

                group_conf.quiz_patterns[quiz_obj.quiz_id] = {
                    "navigation_sequence": leader_nav_seq,
                    "answer_sequence": leader_answers
                }

    def generate_sessions_and_attempts(self):
        self.pre_generate_group_quiz_patterns()

        for user_intern_obj in self.users_internal:
            for quiz_obj in self.quizzes:
                attempt_start_time_dt = datetime.fromtimestamp(quiz_obj.timeopen) + timedelta(
                    seconds=random.randint(0, max(0, int((quiz_obj.timeclose - quiz_obj.timeopen) * 0.1))) # Start within first 10% of quiz window
                )
                
                session_obj = Session(
                    session_id=self.next_id["session"], user_id=user_intern_obj.id,
                    timecreated=int(attempt_start_time_dt.timestamp() - random.randint(60,300)), # Session starts before attempt
                    lastip=fake.ipv4(), sessdata=f"session_data_for_user_{user_intern_obj.id}"
                )
                self.sessions.append(session_obj)
                self.next_id["session"] +=1

                current_quiz_attempt_pk = self.next_id["quiz_attempt"]
                current_question_usage_pk = self.next_id["question_usage"]
                self.next_id["quiz_attempt"] += 1
                self.next_id["question_usage"] += 1

                # In Moodle, context_id often relates to the course module or quiz itself.
                # For simplicity, using course_id as part of a dummy context_id.
                dummy_context_id = quiz_obj.course * 10000 + quiz_obj.quiz_id 
                q_usage_obj = QuestionUsage(question_usage_id=current_question_usage_pk, context_id=dummy_context_id)
                self.question_usages.append(q_usage_obj)

                group_conf_for_user = next((g for g in self.cheating_groups_config if user_intern_obj.id in g.members), None)
                is_user_cheater = group_conf_for_user is not None
                
                question_ids_for_this_quiz = [q.id for q in self.questions_internal if q.quizid == quiz_obj.quiz_id]
                correct_answers_map_for_quiz = self.get_correct_answers_for_quiz(quiz_obj.quiz_id)

                if is_user_cheater and quiz_obj.quiz_id in group_conf_for_user.quiz_patterns:
                    member_idx = group_conf_for_user.members.index(user_intern_obj.id)
                    # Apply start delay based on member index in group (leader starts first)
                    attempt_start_time_dt += timedelta(minutes=group_conf_for_user.patterns["timing"]["start_delay"] * member_idx)
                    
                    leader_patterns = group_conf_for_user.quiz_patterns[quiz_obj.quiz_id]
                    actual_nav_seq = self.generate_navigation_sequence(
                        question_ids_for_this_quiz, is_cheater=True,
                        pattern_config=group_conf_for_user.patterns,
                        base_sequence=leader_patterns["navigation_sequence"]
                    )
                    actual_ans_sel_seq = self.generate_answer_selection_sequence(
                        actual_nav_seq, correct_answers_map_for_quiz, is_cheater=True,
                        pattern_config=group_conf_for_user.patterns,
                        base_answer_selections=leader_patterns["answer_sequence"]
                    )
                    completion_speed_setting = group_conf_for_user.patterns["timing"]["completion_speed"]
                    timing_variance_setting = group_conf_for_user.patterns["timing"]["variance"]
                else: # Honest User
                    actual_nav_seq = self.generate_navigation_sequence(question_ids_for_this_quiz, is_cheater=False)
                    actual_ans_sel_seq = self.generate_answer_selection_sequence(actual_nav_seq, correct_answers_map_for_quiz, is_cheater=False)
                    completion_speed_setting = random.choice(["normal", "slow"]) # Honest users vary
                    timing_variance_setting = random.randint(20, 60) # Higher variance for honest timing

                total_attempt_duration_seconds = 0
                step_timestamp_tracker = attempt_start_time_dt
                
                # --- Generate QuestionAttemptReal and QuestionAttemptStep records ---
                # Map unique question IDs encountered in nav_seq to their QuestionAttemptReal PK
                map_q_id_to_q_attempt_real_pk = {} 
                
                step_sequence_number_counter = 0 # sequencenumber for mdl_question_attempt_steps

                for i_nav_step, current_q_id_in_nav in enumerate(actual_nav_seq):
                    step_sequence_number_counter += 1 # Increment for each step in navigation

                    # Get or create QuestionAttemptReal PK for this question_id in this quiz_usage
                    if current_q_id_in_nav not in map_q_id_to_q_attempt_real_pk:
                        current_q_attempt_real_pk = self.next_id["question_attempt_real"]
                        self.next_id["question_attempt_real"] += 1
                        map_q_id_to_q_attempt_real_pk[current_q_id_in_nav] = current_q_attempt_real_pk
                        
                        q_metadata = next((q for q in self.questions_internal if q.id == current_q_id_in_nav), None)
                        max_mark_val = q_metadata.defaultmark if q_metadata else 1.0
                        
                        q_attempt_real_obj = QuestionAttemptReal(
                            question_attempt_id=current_q_attempt_real_pk,
                            question_usage_id=current_question_usage_pk,
                            questionid=current_q_id_in_nav,
                            maxmark=max_mark_val
                        )
                        self.question_attempts_real.append(q_attempt_real_obj)
                    else:
                        current_q_attempt_real_pk = map_q_id_to_q_attempt_real_pk[current_q_id_in_nav]

                    # Timing for this specific step (viewing/answering a question)
                    if completion_speed_setting == "fast":
                        time_spent_this_step = random.randint(max(1, 5 - timing_variance_setting//2), 20 + timing_variance_setting//2)
                    elif completion_speed_setting == "medium":
                        time_spent_this_step = random.randint(max(1, 15 - timing_variance_setting//2), 45 + timing_variance_setting//2)
                    else: # normal or slow
                        time_spent_this_step = random.randint(max(1, 30 - timing_variance_setting//2), 90 + timing_variance_setting//2)
                    time_spent_this_step = max(1, int(time_spent_this_step))
                    total_attempt_duration_seconds += time_spent_this_step
                    
                    step_timestamp_tracker += timedelta(seconds=time_spent_this_step)

                    q_step_pk = self.next_id["question_attempt_step"]
                    self.next_id["question_attempt_step"] +=1
                    q_attempt_step_obj = QuestionAttemptStep(
                        question_step_id=q_step_pk,
                        question_attempt_id=current_q_attempt_real_pk,
                        sequencenumber=step_sequence_number_counter, # This is the overall step number
                        state="mangrfinished", # Moodle state meaning "Manually graded, marked as finished" or similar
                        timecreated=int(step_timestamp_tracker.timestamp())
                    )
                    self.question_attempt_steps.append(q_attempt_step_obj)

                    step_data_pk = self.next_id["step_data"]
                    self.next_id["step_data"] +=1
                    step_data_obj = QuestionAttemptStepData(
                        step_data_id=step_data_pk,
                        question_step_id=q_step_pk,
                        name="answer", # Could also be other Moodle internal names like "-sequencecheck"
                        value=str(actual_ans_sel_seq[i_nav_step])
                    )
                    self.question_attempt_step_data.append(step_data_obj)
                
                attempt_finish_time_dt = attempt_start_time_dt + timedelta(seconds=total_attempt_duration_seconds)
                # Ensure finish time is within quiz window and after start time
                attempt_finish_time_dt = min(attempt_finish_time_dt, datetime.fromtimestamp(quiz_obj.timeclose - random.randint(1,60)))
                attempt_finish_time_dt = max(attempt_finish_time_dt, attempt_start_time_dt + timedelta(seconds=max(1, len(actual_nav_seq)))) # Min 1 sec per nav step

                # Calculate sumgrades based on the *last* answer for each unique question
                final_score = 0
                total_possible_score = 0
                # Get unique questions in the order they were *last effectively answered* for grading
                # This means finding the last step for each unique question in the nav_seq
                
                # Determine score from the answers given
                answered_question_ids = set() # To score each question only once
                temp_score = 0
                temp_max_score = 0

                for q_id_nav_item in actual_nav_seq: # Iterate through navigation to find last answers
                    if q_id_nav_item not in answered_question_ids:
                         q_meta_info = next((q for q in self.questions_internal if q.id == q_id_nav_item), None)
                         if q_meta_info:
                             temp_max_score += q_meta_info.defaultmark
                         answered_question_ids.add(q_id_nav_item)

                # Find last given answer for each question ID that was part of this quiz
                for q_id_in_quiz in question_ids_for_this_quiz:
                    last_answer_for_this_q_idx = None
                    indices_of_this_q_in_nav = [idx for idx, qid_step in enumerate(actual_nav_seq) if qid_step == q_id_in_quiz]
                    if indices_of_this_q_in_nav:
                        last_occurrence_index = indices_of_this_q_in_nav[-1]
                        last_answer_for_this_q_idx = actual_ans_sel_seq[last_occurrence_index]

                        q_meta_info = next((q for q in self.questions_internal if q.id == q_id_in_quiz), None)
                        if q_meta_info:
                            correct_ans_idx = correct_answers_map_for_quiz.get(q_id_in_quiz, -1)
                            if last_answer_for_this_q_idx == correct_ans_idx:
                                temp_score += q_meta_info.defaultmark
                            # else: # Moodle might apply penalties here based on question settings
                                # temp_score -= (q_meta_info.penalty * q_meta_info.defaultmark)
                                # For simplicity, we only add score for correct, no negative for wrong for sumgrades.
                                # Moodle's sumgrades is typically sum of positive scores.
                
                sumgrades_val_final = round(max(0, temp_score), 4) # Ensure non-negative, round to typical Moodle precision

                quiz_attempt_obj = QuizAttempt(
                    attempt_id=current_quiz_attempt_pk, quiz_id=quiz_obj.quiz_id, user_id=user_intern_obj.id,
                    question_usage_id=current_question_usage_pk,
                    timestart=int(attempt_start_time_dt.timestamp()), timefinish=int(attempt_finish_time_dt.timestamp()),
                    state="finished", sumgrades=sumgrades_val_final
                )
                self.quiz_attempts.append(quiz_attempt_obj)
                
                quiz_grade_obj = QuizGrade(
                    quiz_grades_id=self.next_id["quiz_grade"], quiz_id=quiz_obj.quiz_id,
                    user_id=user_intern_obj.id, final_grade=sumgrades_val_final 
                )
                self.quiz_grades.append(quiz_grade_obj)
                self.next_id["quiz_grade"] += 1
        
        print(f"Generated {len(self.sessions)} sessions, {len(self.quiz_attempts)} quiz attempts.")
        print(f"Generated {len(self.question_usages)} question usages.")
        print(f"Generated {len(self.question_attempts_real)} question_attempts (question-specific).")
        print(f"Generated {len(self.question_attempt_steps)} attempt_steps and {len(self.question_attempt_step_data)} step_data entries.")
        print(f"Generated {len(self.quiz_grades)} quiz_grades.")

    def generate_data(self):
        print("Starting Moodle log generation (Refined)...")
        self.generate_users()
        self.generate_quizzes()
        self.generate_questions_and_answers()
        self.generate_sessions_and_attempts()
        print("Log generation complete!")

    def write_to_csv(self):
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        
        def write_objects_to_csv_custom(objects, filename): # Simplified writer
            if not objects: return
            dict_list = [asdict(obj) for obj in objects]
            if not dict_list: return
            csv_fieldnames = list(dict_list[0].keys())
            with open(os.path.join(self.config["output_dir"], filename), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writeheader()
                writer.writerows(dict_list)
        
        users_for_log_csv = [{ "user_id": u.id, "username": u.username, "firstname": u.firstname, 
                               "lastname": u.lastname, "email": u.email, "lastaccess": u.lastaccess} 
                             for u in self.users_internal] # Filtered user fields for log

        # Output names must match real tables
        write_objects_to_csv_custom(users_for_log_csv, 'mdl_user.csv') # Write only non-ML fields for this
        write_objects_to_csv_custom(self.quizzes, 'mdl_quiz.csv')
        write_objects_to_csv_custom(self.question_answers, 'mdl_question_answers.csv')
        write_objects_to_csv_custom(self.sessions, 'mdl_sessions.csv')
        # The real data has two files named mdl_quiz_attempts.csv and mdl_question_attempts.csv
        # The first one (quiz_attempts) is general attempt info.
        # The second one (question_attempts) is question-specific info within an attempt.
        write_objects_to_csv_custom(self.quiz_attempts, 'mdl_quiz_attempts.csv') # This is the general one
        write_objects_to_csv_custom(self.question_usages, 'mdl_question_usages.csv')
        write_objects_to_csv_custom(self.question_attempts_real, 'mdl_question_attempts.csv') # This is the question-specific one
        write_objects_to_csv_custom(self.question_attempt_steps, 'mdl_question_attempt_steps.csv')
        write_objects_to_csv_custom(self.question_attempt_step_data, 'mdl_question_attempt_step_data.csv')
        write_objects_to_csv_custom(self.quiz_grades, 'mdl_quiz_grades.csv')
        
        self.write_ground_truth_csv()
        self.write_ground_truth_md_and_visualization()
        
        print(f"All data written to {self.config['output_dir']} directory")

    def write_ground_truth_csv(self):
        gt_data = [{'user_id': u.id, 'is_cheater': 1 if u.is_cheater else 0,
                    'cheating_group_id': u.cheating_group_id or 'N/A',
                    'cheating_severity': u.cheating_severity or 'N/A'}
                   for u in self.users_internal]
        
        with open(os.path.join(self.config["output_dir"], 'cheating_ground_truth.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['user_id', 'is_cheater', 'cheating_group_id', 'cheating_severity'])
            writer.writeheader()
            writer.writerows(gt_data)
        print(f"Ground truth CSV written to {self.config['output_dir']}/cheating_ground_truth.csv")

    def write_ground_truth_md_and_visualization(self):
        output_path = self.config["output_dir"]
        
        # --- cheating_ground_truth.md ---
        with open(os.path.join(output_path, 'cheating_ground_truth.md'), 'w') as f:
            f.write("# Cheating Ground Truth (Analysis & Configuration)\n\n")
            f.write("This file contains the ground truth about cheating groups for validation and ML training, based on the generator's configuration.\n\n")
            f.write("## Cheating Groups Configuration\n\n")
            f.write("| Group ID             | Severity        | Member User IDs | Nav Similarity | Nav Noise | Timing Start Delay (min/member) | Timing Variance (s) | Answer Similarity | Wrong Answer Bias |\n")
            f.write("|----------------------|-----------------|-----------------|----------------|-----------|---------------------------------|---------------------|-------------------|-------------------|\n")
            for group_conf in self.cheating_groups_config:
                patterns = group_conf.patterns
                f.write(f"| {group_conf.id:<20} | {group_conf.severity:<15} | {str(group_conf.members):<15} | "
                        f"{patterns['navigation']['similarity']:<14.2f} | {patterns['navigation']['noise']:<9.2f} | "
                        f"{patterns['timing']['start_delay']:<31} | {patterns['timing']['variance']:<19} | "
                        f"{patterns['answers']['similarity']:<17.2f} | {patterns['answers']['wrong_bias']:<17.2f} |\n")
            f.write("\n\n")
        print(f"Ground truth MD analysis file written to {output_path}/cheating_ground_truth.md")

        # --- cheating_visualization.md ---
        with open(os.path.join(output_path, 'cheating_visualization.md'), 'w') as f:
            f.write("# Quiz Attempt Visualization (Generated Data Patterns)\n\n")
            f.write("This file provides a visual representation of generated quiz attempts to help identify suspicious patterns based on the simulation.\n\n")

            for quiz_obj in self.quizzes:
                f.write(f"## Quiz: {quiz_obj.quiz_name} (ID: {quiz_obj.quiz_id})\n\n")
                
                quiz_attempts_for_this_quiz = [qa for qa in self.quiz_attempts if qa.quiz_id == quiz_obj.quiz_id]
                # Sort by group then user_id for clearer visualization
                quiz_attempts_for_this_quiz.sort(key=lambda qa: (
                    next((u.cheating_group_id for u in self.users_internal if u.id == qa.user_id), "Z_Honest"), # Sort honest last
                    qa.user_id
                ))

                # 1. Navigation Patterns & Revisit Analysis (Simplified)
                f.write("### 1. Navigation Patterns (First 15 steps, Question IDs)\n")
                f.write("| User ID | Group ID             | Nav Sequence (Q IDs)         |\n")
                f.write("|---------|----------------------|------------------------------|\n")
                for qa in quiz_attempts_for_this_quiz:
                    user_info = next((u for u in self.users_internal if u.id == qa.user_id), None)
                    group_id_str = user_info.cheating_group_id if user_info and user_info.is_cheater else "Honest"
                    
                    # Reconstruct nav sequence for this attempt
                    # This requires linking QuizAttempt -> QuestionUsage -> QuestionAttemptReal -> QuestionAttemptStep -> Question.id
                    q_usage = next((qu for qu in self.question_usages if qu.question_usage_id == qa.question_usage_id), None)
                    if not q_usage: continue

                    q_attempts_real_for_usage = [qar for qar in self.question_attempts_real if qar.question_usage_id == q_usage.question_usage_id]
                    
                    steps_for_attempt = []
                    for qar_real in q_attempts_real_for_usage:
                        steps_for_qar = [step for step in self.question_attempt_steps if step.question_attempt_id == qar_real.question_attempt_id]
                        steps_for_attempt.extend(steps_for_qar)
                    
                    steps_for_attempt.sort(key=lambda s: s.timecreated) # Overall order of steps
                    
                    # Extract question IDs from steps (from linked QuestionAttemptReal)
                    nav_q_ids = []
                    for step_vis in steps_for_attempt:
                        qar_vis = next((qar for qar in self.question_attempts_real if qar.question_attempt_id == step_vis.question_attempt_id), None)
                        if qar_vis:
                             # Only add if it's a new question or to show sequence, avoid over-complicating with step types for this viz
                            nav_q_ids.append(qar_vis.questionid) 
                    
                    # Consolidate consecutive identical question IDs if they represent single interaction with a question before moving.
                    # For this visualization, let's just show the sequence of question IDs as they were "stepped through".
                    # True navigation reconstruction is complex from step logs.
                    # The `actual_nav_seq` from generation is better.
                    # For now, let's use a simpler representation if reconstruction is too hard:
                    # For this visualization, we'll rely on the fact that `generate_sessions_and_attempts`
                    # creates `QuestionAttemptStep` records in sequence.
                    
                    # Simplified nav sequence from steps (order of unique questions first visited)
                    # This is not the full nav_seq with revisits easily, but an approximation
                    temp_nav_display = []
                    seen_q_for_nav_display = set()
                    for step_disp in steps_for_attempt:
                        qar_disp = next((qar_n for qar_n in self.question_attempts_real if qar_n.question_attempt_id == step_disp.question_attempt_id),None)
                        if qar_disp and qar_disp.questionid not in seen_q_for_nav_display:
                            temp_nav_display.append(str(qar_disp.questionid))
                            seen_q_for_nav_display.add(qar_disp.questionid)
                        if len(temp_nav_display) >=15: break


                    f.write(f"| {qa.user_id:<7} | {group_id_str:<20} | {', '.join(temp_nav_display[:15]):<28} |\n")
                f.write("\n")

                # 2. Answer Patterns (Correct/Incorrect for first 15 unique questions)
                f.write("### 2. Answer Patterns (C=Correct, X=Incorrect for first 15 unique questions)\n")
                f.write("| User ID | Group ID             | Answer Pattern (C/X)       |\n")
                f.write("|---------|----------------------|----------------------------|\n")
                correct_ans_map = self.get_correct_answers_for_quiz(quiz_obj.quiz_id)
                for qa in quiz_attempts_for_this_quiz:
                    user_info = next((u for u in self.users_internal if u.id == qa.user_id), None)
                    group_id_str = user_info.cheating_group_id if user_info and user_info.is_cheater else "Honest"
                    
                    ans_pattern_str = []
                    # Get unique questions answered by this user in this attempt
                    q_usage = next((qu for qu in self.question_usages if qu.question_usage_id == qa.question_usage_id), None)
                    if not q_usage: continue
                    
                    q_attempts_real_for_usage = [qar for qar in self.question_attempts_real if qar.question_usage_id == q_usage.question_usage_id]
                    
                    # Get last answer for each question_id in q_attempts_real_for_usage
                    unique_questions_in_attempt = sorted(list(set(qar.questionid for qar in q_attempts_real_for_usage)))

                    for q_id_unique in unique_questions_in_attempt[:15]: # Limit to 15 for display
                        qar_for_q_id_list = [qar for qar in q_attempts_real_for_usage if qar.questionid == q_id_unique]
                        if not qar_for_q_id_list: continue
                        # Get last step data for this question
                        last_step_for_q = None
                        for qar_entry in reversed(qar_for_q_id_list): # Check all QuestionAttemptReal if multiple for same qid (shouldn't be)
                            steps_for_this_qar = sorted([s for s in self.question_attempt_steps if s.question_attempt_id == qar_entry.question_attempt_id], key=lambda s: s.timecreated)
                            if steps_for_this_qar:
                                last_step_of_qar = steps_for_this_qar[-1]
                                step_data_for_last_step = [sd for sd in self.question_attempt_step_data if sd.question_step_id == last_step_of_qar.question_step_id and sd.name == "answer"]
                                if step_data_for_last_step:
                                    last_step_for_q = step_data_for_last_step[-1] # Should be only one 'answer' per step
                                    break # Found last answer for this question_id

                        if last_step_for_q:
                            user_ans_idx = int(last_step_for_q.value)
                            correct_idx = correct_ans_map.get(q_id_unique, -1)
                            ans_pattern_str.append("C" if user_ans_idx == correct_idx else "X")
                        else:
                            ans_pattern_str.append("?") # No answer found for this question
                    
                    f.write(f"| {qa.user_id:<7} | {group_id_str:<20} | {' '.join(ans_pattern_str):<26} |\n")
                f.write("\n")

                # 3. Timing Patterns
                f.write("### 3. Timing Patterns (Overall)\n")
                f.write("| User ID | Group ID             | Start Time          | Duration (s) | Avg Time/Q (s) |\n")
                f.write("|---------|----------------------|---------------------|--------------|----------------|\n")
                num_unique_questions_in_quiz = len(set(q.id for q in self.questions_internal if q.quizid == quiz_obj.quiz_id))
                for qa in quiz_attempts_for_this_quiz:
                    user_info = next((u for u in self.users_internal if u.id == qa.user_id), None)
                    group_id_str = user_info.cheating_group_id if user_info and user_info.is_cheater else "Honest"
                    start_str = datetime.fromtimestamp(qa.timestart).strftime('%H:%M:%S')
                    duration = qa.timefinish - qa.timestart
                    avg_time_q = duration / num_unique_questions_in_quiz if num_unique_questions_in_quiz > 0 else 0
                    f.write(f"| {qa.user_id:<7} | {group_id_str:<20} | {start_str:<19} | {duration:<12} | {avg_time_q:<14.2f} |\n")
                f.write("\n")

            # Interpretation Guide (can be expanded)
            f.write("## Interpretation Guide\n\n")
            f.write("- **Navigation Sequences**: Look for highly similar sequences of question IDs among users in the same group.\n")
            f.write("- **Answer Patterns**: Identical C/X patterns, especially identical 'X's (wrong answers), are strong indicators of collusion.\n")
            f.write("- **Timing Patterns**: Users in cheating groups might exhibit very similar start times (offset by configured delay), total durations, or suspiciously consistent average time per question. Low variance in timing within a group is key.\n\n")
        print(f"Visualization MD file written to {output_path}/cheating_visualization.md")


    def save_config(self):
        # Ensure output_dir exists
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        with open(os.path.join(self.config["output_dir"], 'generator_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Generator configuration saved to {self.config['output_dir']}/generator_config.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic Moodle logs with cheating patterns (Refined & Visualized)')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG["output_dir"], help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})')
    args = parser.parse_args()
    
    generator = MoodleLogGenerator(args.config)
    if args.output:
        # Convert relative path to absolute if needed
        if not os.path.isabs(args.output):
            args.output = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
        generator.config["output_dir"] = args.output # Override default if provided
    
    generator.generate_data()
    generator.write_to_csv() # This now calls write_ground_truth_md_and_visualization
    generator.save_config()
    
    print(f"All processes complete. Output is in: {generator.config['output_dir']}")

if __name__ == "__main__":
    main()