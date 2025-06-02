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
  "total_quizzes": 2, # Kept low for manageable viz output during refinement
  "questions_per_quiz": 10, # Kept low for manageable viz output during refinement
  "base_date": "2025-03-01 09:00:00",
  "cheating_groups": {
    "high_severity": {
      "count": 1,
      "size": 3,
      "patterns": {
        "navigation": {
          "similarity": 0.96,
          "noise": 0.07
        },
        "timing": {
          "start_delay": 1, # Small delay for testing
          "variance": 2,    # Low variance for high severity
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
          "start_delay": 5,
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
      "count": 1,
      "size": 5,
      "patterns": {
        "navigation": {
          "similarity": 0.55,
          "noise": 0.45
        },
        "timing": {
          "start_delay": 10,
          "variance": 30,
          "completion_speed": "slow" 
        },
        "answers": {
          "similarity": 0.55,
          "wrong_bias": 0.4
        }
      }
    }
  },
  "output_format": "csv",
  "output_dir": "data/moodle_logs_final_viz_corrected", 
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
class User: # Internal representation
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
class Quiz: # For mdl_quiz.csv
    quiz_id: int
    course: int
    quiz_name: str
    timeopen: int
    timeclose: int
    timelimit: int

@dataclass
class Question: # Internal representation for question properties
    id: int
    qtype: str
    name: str
    questiontext: str
    defaultmark: float
    penalty: float
    quizid: int

@dataclass
class QuestionAnswer: # For mdl_question_answers.csv
    question_answers_id: int
    questionid: int 
    answer_text: str
    fraction: float

@dataclass
class Session: # For mdl_sessions.csv
    session_id: int
    user_id: int 
    timecreated: int
    lastip: str
    sessdata: str = ""

@dataclass
class QuizAttempt: # For mdl_quiz_attempts.csv
    attempt_id: int
    quiz_id: int 
    user_id: int 
    question_usage_id: int
    timestart: int
    timefinish: int
    state: str
    sumgrades: float

@dataclass
class QuestionUsage: # For mdl_question_usages.csv
    question_usage_id: int
    context_id: int

@dataclass
class QuestionAttemptReal: # For mdl_question_attempts.csv (question-specific log)
    question_attempt_id: int 
    question_usage_id: int   
    questionid: int          
    maxmark: float

@dataclass
class QuestionAttemptStep: # For mdl_question_attempt_steps.csv
    question_step_id: int 
    question_attempt_id: int 
    sequencenumber: int
    state: str
    timecreated: int

@dataclass
class QuestionAttemptStepData: # For mdl_question_attempt_step_data.csv
    step_data_id: int 
    question_step_id: int 
    name: str
    value: str

@dataclass
class QuizGrade: # For mdl_quiz_grades.csv
    quiz_grades_id: int 
    quiz_id: int
    user_id: int
    final_grade: float

@dataclass
class AttemptVisualizationData: # New dataclass to hold detailed info for visualization
    user_id: int
    quiz_id: int
    quiz_name: str
    attempt_id: int
    is_cheater: bool
    cheating_group_id: Optional[str]
    timestart: int
    timefinish: int
    actual_nav_seq: List[int] 
    actual_ans_sel_seq: List[int] 
    step_event_details: List[Dict[str, Any]] # List of {'q_id': int, 'view_timestamp': int, 'answer_timestamp': int, 'duration_on_q_interaction': int}
    sumgrades: float


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
        self.all_attempts_visualization_details: List[AttemptVisualizationData] = []
        
        self.next_id = {
            "user": 1, "quiz": 1, "question_internal": 1, "Youtube": 1, # Corrected "Youtube" to "Youtube"
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
                group = CheatingGroupConfig(id=group_name, severity=severity, members=members, patterns=config_details["patterns"])
                self.cheating_groups_config.append(group)
                self.cheater_user_ids.extend(members)
        if len(self.cheater_user_ids) > total_users_in_config:
             raise ValueError("Assigned more cheaters than total users. Adjust config.")
        print(f"Created {len(self.cheating_groups_config)} cheating groups with {len(self.cheater_user_ids)} total cheaters.")
        print(f"Remaining {total_users_in_config - len(self.cheater_user_ids)} users will be honest.")
    
    def generate_users(self):
        for user_id_val in range(1, self.config["total_users"] + 1):
            is_cheater_flag = user_id_val in self.cheater_user_ids
            cheating_group_identifier = None; cheating_sev = None
            if is_cheater_flag:
                for group_conf in self.cheating_groups_config:
                    if user_id_val in group_conf.members:
                        cheating_group_identifier = group_conf.id; cheating_sev = group_conf.severity; break
            user_obj = User(
                id=user_id_val, username=fake.unique.user_name(), password=fake.sha256(),
                firstname=fake.first_name(), lastname=fake.last_name(), email=fake.unique.email(),
                lastaccess=int((self.base_date + timedelta(days=random.randint(-30,0))).timestamp()),
                is_cheater=is_cheater_flag, cheating_group_id=cheating_group_identifier, cheating_severity=cheating_sev)
            self.users_internal.append(user_obj)
        print(f"Generated {len(self.users_internal)} internal user representations")
    
    def generate_quiz_topics(self):
        topics = ["Intro to Python", "Advanced Algorithms", "DB Management", "Network Security", "OS Concepts", 
                  "Software Design Patterns", "AI Ethics", "ML Foundations", "3D Rendering", "Full-Stack Web Dev"]
        return random.sample(topics, min(len(topics), self.config["total_quizzes"]))
    
    def generate_quizzes(self):
        topics = self.generate_quiz_topics()
        for i in range(1, self.config["total_quizzes"] + 1):
            topic = topics[i-1] if i <= len(topics) else f"Quiz Topic {i}"
            timeopen_dt = self.base_date + timedelta(days=(i-1)*7 + random.randint(0,1), hours=random.randint(8,10))
            timeclose_dt = timeopen_dt + timedelta(days=random.randint(1,3), hours=random.randint(1,12))
            quiz_obj = Quiz(
                quiz_id=self.next_id["quiz"], course=random.randint(101, 103), 
                quiz_name=f"Quiz {i}: {topic}", timeopen=int(timeopen_dt.timestamp()),
                timeclose=int(timeclose_dt.timestamp()), timelimit=random.choice([1800, 2700, 3600, 5400, 7200]))
            self.quizzes.append(quiz_obj); self.next_id["quiz"] += 1
        print(f"Generated {len(self.quizzes)} quizzes")
    
    def generate_questions_and_answers(self):
        for quiz_obj in self.quizzes:
            for q_num in range(1, self.config["questions_per_quiz"] + 1):
                q_id = self.next_id["question_internal"]
                question_obj = Question(
                    id=q_id, qtype="multichoice", name=f"Q{q_num}",
                    questiontext=f"Content of Question {q_num} for {quiz_obj.quiz_name}",
                    defaultmark=round(random.uniform(1.0, 2.0),1), 
                    penalty=round(random.uniform(0.0, 0.33), 2), quizid=quiz_obj.quiz_id)
                self.questions_internal.append(question_obj); self.next_id["question_internal"] += 1
                options = [f"Option {chr(65+opt_idx)} for Q{q_num} ({quiz_obj.quiz_name})" for opt_idx in range(4)]
                random.shuffle(options); correct_answer_idx = random.randint(0, 3)
                for ans_idx in range(4):
                    ans_id = self.next_id["Youtube"] # Corrected key from "Youtube"
                    answer_obj = QuestionAnswer(question_answers_id=ans_id, questionid=q_id, answer_text=options[ans_idx],
                                                fraction=1.0 if ans_idx == correct_answer_idx else 0.0)
                    self.question_answers.append(answer_obj); self.next_id["Youtube"] += 1
        print(f"Generated {len(self.questions_internal)} internal questions and {len(self.question_answers)} answers")

    def get_correct_answers_for_quiz(self, quiz_id_val: int) -> Dict[int, int]:
        correct_answers_map = {} 
        questions_for_this_quiz = [q for q in self.questions_internal if q.quizid == quiz_id_val]
        for question_intern in questions_for_this_quiz:
            answers_for_this_q = sorted([a for a in self.question_answers if a.questionid == question_intern.id], key=lambda x: x.question_answers_id)
            try:
                correct_choice_idx = next((i for i, a in enumerate(answers_for_this_q) if a.fraction >= 1.0), 0)
                correct_answers_map[question_intern.id] = correct_choice_idx
            except StopIteration:
                 correct_answers_map[question_intern.id] = 0
                 print(f"Warning: No correct answer found for QID {question_intern.id} in quiz {quiz_id_val}. Defaulting to option 0.")
        return correct_answers_map

    def generate_navigation_sequence(self, question_ids_in_quiz: List[int], is_cheater=False, pattern_config=None, base_sequence: Optional[List[int]]=None):
        num_questions = len(question_ids_in_quiz)
        if not question_ids_in_quiz: return []
        sequence = []
        if is_cheater and base_sequence and pattern_config:
            sequence = base_sequence.copy()
            noise = pattern_config["navigation"]["noise"]
            swaps = int(num_questions * noise) 
            for _ in range(swaps):
                if num_questions > 1: i, j = random.sample(range(num_questions), 2); sequence[i], sequence[j] = sequence[j], sequence[i]
        else:
            sequence = random.sample(question_ids_in_quiz, len(question_ids_in_quiz))
        revisit_count_factor = 0.35 if is_cheater else 0.5 # Slightly increased revisit for more complex paths
        revisit_count = random.randint(0, int(num_questions * revisit_count_factor))
        
        for _ in range(revisit_count):
            if not sequence: continue
            question_to_revisit = random.choice(sequence)
            try:
                indices = [k for k, x in enumerate(sequence) if x == question_to_revisit]
                first_pos = indices[0] if indices else 0 
                insert_pos = random.randint(first_pos + 1, len(sequence))
                sequence.insert(insert_pos, question_to_revisit)
            except (ValueError, IndexError):
                pass
        
        return sequence
    
    def generate_answer_selection_sequence(self, nav_sequence: List[int], correct_answers_map: Dict[int, int], 
                                           is_cheater=False, pattern_config=None, base_answer_selections: Optional[List[int]]=None):
        actual_answers_selected = []
        if is_cheater and base_answer_selections and pattern_config:
            ans_similarity = pattern_config["answers"]["similarity"]; wrong_bias = pattern_config["answers"]["wrong_bias"]
            for i, q_id in enumerate(nav_sequence):
                correct_ans_idx = correct_answers_map.get(q_id, 0)
                if i < len(base_answer_selections):
                    leader_answer = base_answer_selections[i]
                    if random.random() < ans_similarity: actual_answers_selected.append(leader_answer)
                    elif leader_answer != correct_ans_idx and random.random() < wrong_bias: actual_answers_selected.append(leader_answer)
                    else: 
                        actual_answers_selected.append(correct_ans_idx if random.random() < 0.6 else random.choice([opt for opt in range(4) if opt != correct_ans_idx] or [correct_ans_idx]))
                else: 
                    actual_answers_selected.append(correct_ans_idx if random.random() < 0.5 else random.choice([opt for opt in range(4) if opt != correct_ans_idx] or [correct_ans_idx]))
        else:  # Honest User
            for q_id in nav_sequence:
                correct_ans_idx = correct_answers_map.get(q_id, 0)
                actual_answers_selected.append(correct_ans_idx if random.random() < 0.75 else random.choice([opt for opt in range(4) if opt != correct_ans_idx] or [correct_ans_idx]))
        return actual_answers_selected

    def pre_generate_group_quiz_patterns(self):
        for group_conf in self.cheating_groups_config:
            for quiz_obj in self.quizzes:
                question_ids_for_quiz = [q.id for q in self.questions_internal if q.quizid == quiz_obj.quiz_id]
                if not question_ids_for_quiz: continue
                leader_nav_seq = self.generate_navigation_sequence(question_ids_for_quiz, is_cheater=True, pattern_config=group_conf.patterns)
                correct_ans_map = self.get_correct_answers_for_quiz(quiz_obj.quiz_id)
                leader_answers = self.generate_answer_selection_sequence(leader_nav_seq, correct_ans_map, is_cheater=True, pattern_config=group_conf.patterns)
                group_conf.quiz_patterns[quiz_obj.quiz_id] = {"navigation_sequence": leader_nav_seq, "answer_sequence": leader_answers}
    
    def generate_sessions_and_attempts(self):
        self.pre_generate_group_quiz_patterns()
        for user_intern_obj in self.users_internal:
            user_is_cheater_flag = user_intern_obj.is_cheater
            user_cheating_group_id_str = user_intern_obj.cheating_group_id

            for quiz_obj in self.quizzes:
                attempt_start_time_dt = datetime.fromtimestamp(quiz_obj.timeopen) + timedelta(seconds=random.randint(0, max(0, int((quiz_obj.timeclose - quiz_obj.timeopen) * 0.2))))
                session_obj = Session(session_id=self.next_id["session"], user_id=user_intern_obj.id,
                                    timecreated=int(attempt_start_time_dt.timestamp() - random.randint(60,300)),
                                    lastip=fake.ipv4(), sessdata=f"session_uid{user_intern_obj.id}_qid{quiz_obj.quiz_id}")
                self.sessions.append(session_obj); self.next_id["session"] +=1
                
                current_quiz_attempt_pk = self.next_id["quiz_attempt"]; self.next_id["quiz_attempt"] += 1
                current_question_usage_pk = self.next_id["question_usage"]; self.next_id["question_usage"] += 1
                self.question_usages.append(QuestionUsage(question_usage_id=current_question_usage_pk, context_id=(quiz_obj.course * 1000 + quiz_obj.quiz_id)))

                group_conf = next((g for g in self.cheating_groups_config if user_intern_obj.id in g.members), None)
                q_ids_this_quiz = [q.id for q in self.questions_internal if q.quizid == quiz_obj.quiz_id]
                correct_ans_map_this_quiz = self.get_correct_answers_for_quiz(quiz_obj.quiz_id)
                
                # These will store the exact sequences for this attempt for visualization
                attempt_actual_nav_seq: List[int] = []
                attempt_actual_ans_sel_seq: List[int] = []
                attempt_step_event_details: List[Dict[str,Any]] = []

                if group_conf and quiz_obj.quiz_id in group_conf.quiz_patterns: # Cheater
                    member_idx = group_conf.members.index(user_intern_obj.id)
                    attempt_start_time_dt += timedelta(minutes=group_conf.patterns["timing"]["start_delay"] * member_idx)
                    leader_pats = group_conf.quiz_patterns[quiz_obj.quiz_id]
                    attempt_actual_nav_seq = self.generate_navigation_sequence(q_ids_this_quiz, True, group_conf.patterns, leader_pats["navigation_sequence"])
                    attempt_actual_ans_sel_seq = self.generate_answer_selection_sequence(attempt_actual_nav_seq, correct_ans_map_this_quiz, True, group_conf.patterns, leader_pats["answer_sequence"])
                    comp_speed = group_conf.patterns["timing"]["completion_speed"]; time_var = group_conf.patterns["timing"]["variance"]
                else: # Honest
                    attempt_actual_nav_seq = self.generate_navigation_sequence(q_ids_this_quiz, False)
                    attempt_actual_ans_sel_seq = self.generate_answer_selection_sequence(attempt_actual_nav_seq, correct_ans_map_this_quiz, False)
                    comp_speed = random.choice(["normal", "slow", "medium"]); time_var = random.randint(25, 70) 

                current_step_time = attempt_start_time_dt
                map_q_to_q_attempt_real_pk = {} 
                step_seq_counter = 0 

                for i_nav, q_id_nav in enumerate(attempt_actual_nav_seq):
                    step_seq_counter += 1
                    q_real_attempt_pk = map_q_to_q_attempt_real_pk.get(q_id_nav)
                    if not q_real_attempt_pk:
                        q_real_attempt_pk = self.next_id["question_attempt_real"]; self.next_id["question_attempt_real"] += 1
                        map_q_to_q_attempt_real_pk[q_id_nav] = q_real_attempt_pk
                        q_m = next((q for q in self.questions_internal if q.id == q_id_nav), None)
                        self.question_attempts_real.append(QuestionAttemptReal(q_real_attempt_pk, current_question_usage_pk, q_id_nav, (q_m.defaultmark if q_m else 1.0)))
                    
                    view_ts = int(current_step_time.timestamp())
                    if comp_speed == "fast": step_dur = random.randint(max(1, 3 - time_var//3), 15 + time_var//3)
                    elif comp_speed == "medium": step_dur = random.randint(max(1, 10 - time_var//2), 35 + time_var//2)
                    else: step_dur = random.randint(max(1, 20 - time_var//2), 70 + time_var//2)
                    step_dur = max(1, int(step_dur))
                    current_step_time += timedelta(seconds=step_dur)
                    ans_ts = int(current_step_time.timestamp())

                    attempt_step_event_details.append({"q_id": q_id_nav, "view_timestamp": view_ts, "answer_timestamp": ans_ts, "duration_on_q_interaction": step_dur})
                    
                    q_step_pk = self.next_id["question_attempt_step"]; self.next_id["question_attempt_step"] +=1
                    q_ans_idx = attempt_actual_ans_sel_seq[i_nav]
                    q_correct_idx = correct_ans_map_this_quiz.get(q_id_nav)
                    step_state = "gradedright" if q_ans_idx == q_correct_idx else "gradedwrong"
                    self.question_attempt_steps.append(QuestionAttemptStep(q_step_pk, q_real_attempt_pk, step_seq_counter, step_state, ans_ts))
                    
                    step_data_pk = self.next_id["step_data"]; self.next_id["step_data"] +=1
                    self.question_attempt_step_data.append(QuestionAttemptStepData(step_data_pk, q_step_pk, "answer", str(q_ans_idx)))

                attempt_finish_dt = current_step_time
                attempt_finish_dt = min(attempt_finish_dt, datetime.fromtimestamp(quiz_obj.timeclose - random.randint(1,60)))
                attempt_finish_dt = max(attempt_finish_dt, attempt_start_time_dt + timedelta(seconds=max(1, len(attempt_actual_nav_seq))))

                final_sgs = 0.0
                for unique_qid in q_ids_this_quiz:
                    q_m = next((q for q in self.questions_internal if q.id == unique_qid), None)
                    if not q_m: continue
                    last_ans = -1
                    for nav_idx_rev in range(len(attempt_actual_nav_seq) - 1, -1, -1):
                        if attempt_actual_nav_seq[nav_idx_rev] == unique_qid: last_ans = attempt_actual_ans_sel_seq[nav_idx_rev]; break
                    if last_ans != -1 and last_ans == correct_ans_map_this_quiz.get(unique_qid, -2): final_sgs += q_m.defaultmark
                sumgrades_final = round(max(0, final_sgs), 4)

                self.quiz_attempts.append(QuizAttempt(current_quiz_attempt_pk, quiz_obj.quiz_id, user_intern_obj.id, current_question_usage_pk, 
                                                    int(attempt_start_time_dt.timestamp()), int(attempt_finish_dt.timestamp()), "finished", sumgrades_final))
                self.quiz_grades.append(QuizGrade(self.next_id["quiz_grade"], quiz_obj.quiz_id, user_intern_obj.id, sumgrades_final)); self.next_id["quiz_grade"] +=1
                
                self.all_attempts_visualization_details.append(AttemptVisualizationData(
                    user_intern_obj.id, quiz_obj.quiz_id, quiz_obj.quiz_name, current_quiz_attempt_pk, user_is_cheater_flag, 
                    user_cheating_group_id_str, int(attempt_start_time_dt.timestamp()), int(attempt_finish_dt.timestamp()),
                    attempt_actual_nav_seq.copy(), attempt_actual_ans_sel_seq.copy(), attempt_step_event_details.copy(), sumgrades_final))
        # Final print counts
        print(f"Generated {len(self.sessions)} sessions, {len(self.quiz_attempts)} quiz attempts.")
        print(f"Generated {len(self.question_usages)} question usages.")
        print(f"Generated {len(self.question_attempts_real)} question_attempts (question-specific).")
        print(f"Generated {len(self.question_attempt_steps)} attempt_steps and {len(self.question_attempt_step_data)} step_data entries.")
        print(f"Generated {len(self.quiz_grades)} quiz_grades.")
        print(f"Collected {len(self.all_attempts_visualization_details)} detailed entries for visualization.")

    
    def generate_data(self):
        print("Starting Moodle log generation (Finalized Viz Prep)...")
        self.generate_users()
        self.generate_quizzes()
        self.generate_questions_and_answers()
        self.generate_sessions_and_attempts()
        print("Log generation complete!")
    
    def write_to_csv(self):
        if not os.path.exists(self.config["output_dir"]): os.makedirs(self.config["output_dir"])
        def write_objects_to_csv(objects, filename):
            if not objects: print(f"Skipping {filename}, no data."); return
            # Handle both dataclass instances and regular dictionaries
            dict_list = []
            for obj in objects:
                if hasattr(obj, '__dataclass_fields__'):  # It's a dataclass
                    dict_list.append(asdict(obj))
                elif isinstance(obj, dict):  # It's already a dictionary
                    dict_list.append(obj)
                else:
                    raise TypeError(f"Expected dataclass instance or dict, got {type(obj)}")
            
            if not dict_list: print(f"Skipping {filename}, no dict data."); return
            csv_fieldnames = list(dict_list[0].keys())
            filepath = os.path.join(self.config["output_dir"], filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames); writer.writeheader(); writer.writerows(dict_list)
            print(f"Written {len(dict_list)} rows to {filepath}")

        users_log_data = [{"user_id": u.id, "username": u.username, "firstname": u.firstname, "lastname": u.lastname, 
                           "email": u.email, "lastaccess": u.lastaccess} for u in self.users_internal]
        write_objects_to_csv(users_log_data, 'mdl_user.csv')
        write_objects_to_csv(self.quizzes, 'mdl_quiz.csv')
        write_objects_to_csv(self.question_answers, 'mdl_question_answers.csv')
        write_objects_to_csv(self.sessions, 'mdl_sessions.csv')
        write_objects_to_csv(self.quiz_attempts, 'mdl_quiz_attempts.csv') # General attempt info
        write_objects_to_csv(self.question_usages, 'mdl_question_usages.csv')
        write_objects_to_csv(self.question_attempts_real, 'mdl_question_attempts.csv') # Question-specific log
        write_objects_to_csv(self.question_attempt_steps, 'mdl_question_attempt_steps.csv')
        write_objects_to_csv(self.question_attempt_step_data, 'mdl_question_attempt_step_data.csv')
        write_objects_to_csv(self.quiz_grades, 'mdl_quiz_grades.csv')
        
        self.write_ground_truth_csv()
        self.write_ground_truth_md_and_visualization() # Uses self.all_attempts_visualization_details
        print(f"All data logs written to {self.config['output_dir']}")

    def write_ground_truth_csv(self):
        gt_data = [{'user_id': u.id, 'is_cheater': 1 if u.is_cheater else 0,
                    'cheating_group_id': u.cheating_group_id or 'N/A',
                    'cheating_severity': u.cheating_severity or 'N/A'}
                   for u in self.users_internal]
        gt_path = os.path.join(self.config["output_dir"], 'cheating_ground_truth.csv')
        with open(gt_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['user_id', 'is_cheater', 'cheating_group_id', 'cheating_severity'])
            writer.writeheader(); writer.writerows(gt_data)
        print(f"Ground truth CSV written to {gt_path}")

    def write_ground_truth_md_and_visualization(self):
        # --- cheating_ground_truth.md (Configuration Summary) ---
        md_gt_path = os.path.join(self.config["output_dir"], 'cheating_ground_truth.md')
        with open(md_gt_path, 'w') as f:
            f.write("# Cheating Ground Truth (Generator Configuration)\n\n")
            f.write("This file summarizes the configuration parameters used to generate cheating patterns.\n\n")
            f.write("## Cheating Groups Configuration\n\n")
            f.write("| Group ID             | Severity        | Member User IDs     | Nav Similarity | Nav Noise | Timing Start Delay (min/member) | Timing Variance (s) | Completion Speed | Answer Similarity | Wrong Answer Bias |\n")
            f.write("|----------------------|-----------------|---------------------|----------------|-----------|---------------------------------|---------------------|------------------|-------------------|-------------------|\n")
            for group_conf in self.cheating_groups_config:
                p = group_conf.patterns
                f.write(f"| {group_conf.id:<20} | {group_conf.severity:<15} | {str(group_conf.members):<19} | "
                        f"{p['navigation']['similarity']:<14.2f} | {p['navigation']['noise']:<9.2f} | "
                        f"{p['timing']['start_delay']:<31} | {p['timing']['variance']:<19} | "
                        f"{p['timing']['completion_speed']:<16} | {p['answers']['similarity']:<17.2f} | {p['answers']['wrong_bias']:<17.2f} |\n")
            f.write("\n\n")
        print(f"Ground truth MD (configuration) file written to {md_gt_path}")

        # --- cheating_visualization.md (Observed Patterns in Generated Data) ---
        md_viz_path = os.path.join(self.config["output_dir"], 'cheating_visualization.md')
        with open(md_viz_path, 'w') as f:
            f.write("# Quiz Attempt Visualization (Observed Generated Patterns)\n\n")
            f.write("This file visualizes patterns from the generated data, using the detailed attempt information.\n")
            f.write(f"Data generated with seed: {self.config['seed']}\n\n")

            sorted_viz_details = sorted(self.all_attempts_visualization_details, 
                                        key=lambda x: (x.quiz_id, x.cheating_group_id or "Z_Honest", x.user_id))
            
            max_nav_display = 25 # Max navigation steps to show in MD
            max_timing_interactions_display = 15 # Max timing interactions to show

            for quiz_obj_viz in self.quizzes: # Iterate through quizzes to structure the MD
                f.write(f"\n## Quiz: {quiz_obj_viz.quiz_name} (ID: {quiz_obj_viz.quiz_id})\n\n")
                
                attempts_for_this_quiz_viz = [ad for ad in sorted_viz_details if ad.quiz_id == quiz_obj_viz.quiz_id]
                if not attempts_for_this_quiz_viz:
                    f.write("No attempts recorded for this quiz in visualization data.\n\n")
                    continue

                # Section 1: Navigation and Answer Patterns
                f.write("### 1. Navigation & Answer Patterns\n")
                f.write(f"| UserID | Group         | AttemptID | Nav Sequence (Q IDs, first {max_nav_display} steps) | Revisits (QID:Count) | Answer Pattern (C/X, for displayed Nav Seq) |\n")
                f.write( "|--------|---------------|-----------|---------------------------------------|----------------------|-------------------------------------------|\n")
                for ad in attempts_for_this_quiz_viz:
                    group_str = ad.cheating_group_id if ad.is_cheater else "Honest"
                    nav_seq_display = ", ".join(map(str, ad.actual_nav_seq[:max_nav_display]))
                    if len(ad.actual_nav_seq) > max_nav_display: nav_seq_display += f", ... ({len(ad.actual_nav_seq) - max_nav_display} more)"
                    
                    revisit_counts_viz = defaultdict(int)
                    for q_id_nav_viz in ad.actual_nav_seq: revisit_counts_viz[q_id_nav_viz] += 1
                    revisit_str_viz = ", ".join([f"{qid}:{cnt}" for qid, cnt in revisit_counts_viz.items() if cnt > 1]) or "None"
                    
                    correct_ans_map_viz = self.get_correct_answers_for_quiz(ad.quiz_id)
                    ans_pattern_list_viz = []
                    for i_viz, q_id_viz in enumerate(ad.actual_nav_seq[:max_nav_display]):
                        if i_viz < len(ad.actual_ans_sel_seq): # Ensure index exists
                            ans_idx_viz = ad.actual_ans_sel_seq[i_viz]
                            correct_idx_viz = correct_ans_map_viz.get(q_id_viz, -1)
                            ans_pattern_list_viz.append("C" if ans_idx_viz == correct_idx_viz else "X")
                        else: ans_pattern_list_viz.append("?") # Should not happen if data is consistent
                    ans_pattern_display_str_viz = " ".join(ans_pattern_list_viz)
                    f.write(f"| {ad.user_id:<6} | {group_str:<13} | {ad.attempt_id:<9} | {nav_seq_display:<37} | {revisit_str_viz:<20} | {ans_pattern_display_str_viz:<41} |\n")
                f.write("\n")

                # Section 2: Timing Patterns
                f.write(f"### 2. Timing Patterns for Quiz: {quiz_obj_viz.quiz_name} (ID: {quiz_obj_viz.quiz_id})\n")
                f.write(f"| UserID | Group         | AttemptID | Start Time          | Duration (s) | Interactions | Avg Interaction Time (s) | StdDev Interaction Time (s) | Grade | Interactions Detail (QID:Dur, first {max_timing_interactions_display}) |\n")
                f.write( "|--------|---------------|-----------|---------------------|--------------|--------------|--------------------------|-----------------------------|-------|---------------------------------------------------|\n")
                for ad in attempts_for_this_quiz_viz:
                    group_str = ad.cheating_group_id if ad.is_cheater else "Honest"
                    start_time_str = datetime.fromtimestamp(ad.timestart).strftime('%H:%M:%S')
                    duration_total = ad.timefinish - ad.timestart
                    num_interactions = len(ad.step_event_details)
                    interaction_durations = [step['duration_on_q_interaction'] for step in ad.step_event_details]
                    avg_inter_time = np.mean(interaction_durations) if interaction_durations else 0
                    stddev_inter_time = np.std(interaction_durations) if len(interaction_durations) > 1 else 0
                    
                    interactions_detail_str_viz = ", ".join([f"{s['q_id']}:{s['duration_on_q_interaction']}" for s in ad.step_event_details[:max_timing_interactions_display]])
                    if len(ad.step_event_details) > max_timing_interactions_display: interactions_detail_str_viz += ", ..."
                    f.write(f"| {ad.user_id:<6} | {group_str:<13} | {ad.attempt_id:<9} | {start_time_str:<19} | {duration_total:<12} | {num_interactions:<12} | {avg_inter_time:<24.2f} | {stddev_inter_time:<27.2f} | {ad.sumgrades:<5.2f} | {interactions_detail_str_viz:<49} |\n")
                f.write("\n")

                # Section 3: Transition Time Correlation (within cheating groups)
                f.write(f"### 3. Transition Time Correlation Analysis for Quiz: {quiz_obj_viz.quiz_name} (ID: {quiz_obj_viz.quiz_id})\n")
                f.write("Correlation of time intervals between consecutive question interactions for users within the same cheating group.\n")
                f.write("| Group ID             | User Pair (ID1, ID2) | Transition Time Correlation | Num Common Transitions |\n")
                f.write("|----------------------|----------------------|-----------------------------|------------------------|\n")
                
                processed_pairs_for_corr = set()
                for group_conf_viz in self.cheating_groups_config:
                    members_in_group = group_conf_viz.members
                    member_attempts_in_quiz = [att for att in attempts_for_this_quiz_viz if att.user_id in members_in_group]
                    
                    if len(member_attempts_in_quiz) < 2: continue # Need at least two members for correlation

                    for i in range(len(member_attempts_in_quiz)):
                        for j in range(i + 1, len(member_attempts_in_quiz)):
                            att1 = member_attempts_in_quiz[i]
                            att2 = member_attempts_in_quiz[j]
                            
                            pair_key = tuple(sorted((att1.user_id, att2.user_id)))
                            if pair_key in processed_pairs_for_corr: continue
                            processed_pairs_for_corr.add(pair_key)

                            # Extract sequences of timestamps of 'answer_timestamp' (end of interaction for a question)
                            # These represent the points in time when a user moves from one question interaction to the next.
                            ts_seq1 = sorted([step['answer_timestamp'] for step in att1.step_event_details])
                            ts_seq2 = sorted([step['answer_timestamp'] for step in att2.step_event_details])
                            if len(ts_seq1) < 2 or len(ts_seq2) < 2:
                                user_pair_str = f"({att1.user_id}, {att2.user_id})"
                                f.write(f"| {group_conf_viz.id:<20} | {user_pair_str:<20} | Not enough data           | 0                      |\n")
                                continue
                            
                            transitions1 = np.diff(ts_seq1)
                            transitions2 = np.diff(ts_seq2)
                            
                            min_len = min(len(transitions1), len(transitions2))
                            if min_len < 2: # Need at least 2 transition times for correlation
                                user_pair_str = f"({att1.user_id}, {att2.user_id})"
                                f.write(f"| {group_conf_viz.id:<20} | {user_pair_str:<20} | Not enough transitions    | {min_len:<22} |\n")
                                continue
                            
                            corr = np.corrcoef(transitions1[:min_len], transitions2[:min_len])[0, 1]
                            corr_str = f"{corr:.4f}" if not np.isnan(corr) else "NaN"
                            user_pair_str = f"({att1.user_id}, {att2.user_id})"
                            f.write(f"| {group_conf_viz.id:<20} | {user_pair_str:<20} | {corr_str:<27} | {min_len:<22} |\n")
                f.write("\n")

            f.write("\n\n### Interpretation Guide\n\n")
            f.write("- **Navigation Sequences & Answer Patterns**: Cheating groups (especially high/medium severity) should exhibit very similar `Nav Sequence` and `Answer Pattern (C/X)` strings among members. Honest users will be more varied.\n")
            f.write("- **Revisits**: Similar revisit patterns (specific questions revisited similar number of times) can be indicative if coupled with other similarities.\n")
            f.write("- **Timing Patterns**:\n")
            f.write("  - `Start Time`: Cheating group members might show staggered starts based on `Timing Start Delay` config.\n")
            f.write("  - `Duration (s)` & `Avg Interaction Time (s)`: Reflects `Completion Speed`. Lower `StdDev Interaction Time (s)` for cheaters indicates consistent pacing.\n")
            f.write("  - `Interactions Detail`: Look for similar durations on the same sequence of questions among group members.\n")
            f.write("- **Transition Time Correlation**: Values close to 1.0 for user pairs within a cheating group suggest highly synchronized progression through the quiz, a strong indicator of real-time collusion.\n")
            f.write("- **Grade**: While not a sole indicator, consistently high grades among a group with other similarities can be corroborative.\n")

        print(f"Visualization MD file (corrected) written to {md_viz_path}")

    
    def save_config(self):
        if not os.path.exists(self.config["output_dir"]): os.makedirs(self.config["output_dir"])
        cfg_path = os.path.join(self.config["output_dir"], 'generator_config.json')
        with open(cfg_path, 'w') as f: json.dump(self.config, f, indent=2)
        print(f"Generator configuration saved to {cfg_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic Moodle logs with cheating patterns (Finalized Viz Corrected)')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG["output_dir"], help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})')
    args = parser.parse_args()
    
    generator = MoodleLogGenerator(args.config)
    if args.output:
        if not os.path.isabs(args.output): args.output = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
        generator.config["output_dir"] = args.output
    
    generator.generate_data()
    generator.write_to_csv()
    generator.save_config()
    print(f"All processes complete. Output is in: {generator.config['output_dir']}")

if __name__ == "__main__":
    main()