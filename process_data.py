from typing import List
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

class ProcessData:
    def __init__(
            self,
            file_path,
            tokenizer,
            hr_gap:int=5
    ):
        self.file_path = file_path
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.content = file.read().split('\n')[1:]
        self.parsed_logs = self.parse_chat_logs(self.content)
        self.combined_msgs = self.combine_messages(self.parsed_logs)
        self.grouped_msgs = self.group_messages(self.combined_msgs, hr_gap)
        self.get_token_distribution(tokenizer)

    def parse_chat_logs(self, logs: List[str]) -> List[dict]:
        """
        Parse chat logs from WhatsApp.

        Parameters
        ----------
        logs : List[str]
            List of strings where each string is a single log.

        Returns
        -------
        List[dict]
            List of dictionaries where each dictionary contains the fields:
            - date: str
            - time: str
            - name: str
            - message: str
        """
        parsed_logs = []
        for log in logs:
            try:
                date_time, rest = log.split(" - ", 1)
                date, time = date_time.split(", ")
                name_message = rest.split(": ", 1)
                if len(name_message) == 2:
                    name, message = name_message
                    if message != '<Media omitted>':
                        if '<This message was edited>' in message:
                            message = message.replace('<This message was edited>', '')
                        parsed_logs.append({
                            "date": date,
                            "time": time,
                            "name": name.strip(),
                            "message": message.strip()
                        })
            except ValueError:
                continue
        return parsed_logs
    
    @property
    def get_chat_names(self):
        """
        Get all unique chat names from the parsed logs.

        Returns
        -------
        List[str]
            List of unique chat names.
        """
        chat_names = []
        for log in self.parsed_logs:
            name = log['name']
            if name not in chat_names:
                chat_names.append(name)
        return chat_names
    
    @property
    def get_message_counts(self):
        """
        Get a dictionary with the count of messages from each user in the parsed logs.

        Returns
        -------
        Dict[str, int]
            Dictionary with user names as keys and the count of messages as values.
        """
        message_counts = {}
        for group in self.grouped_conversations:
            for message in group:
                name = message['name']
                if name not in message_counts:
                    message_counts[name] = 0
                message_counts[name] += 1
        return message_counts
    
    @property
    def get_conversation_lengths(self):
        """
        Get a list of conversation lengths from the grouped conversations.

        Returns
        -------
        List[int]
            List of conversation lengths.
        """
        conversation_lengths = []
        for group in self.grouped_msgs:
            conversation_lengths.append(len(group))
        return conversation_lengths

    def combine_messages(self, parsed_logs: List[dict]) -> List[dict]:
        """
        Combines consecutive messages from the same user into a single message.

        Parameters
        ----------
        parsed_logs : list of dict
            List of parsed log dictionaries, each containing the keys: 'date', 'time', 'name', and 'message'.

        Returns
        -------
        list of dict
            A list of dictionaries where consecutive messages from the same user are combined into a single message.
        """
        combined_messages = []
        current_message = None

        for log in parsed_logs:
            if current_message is None:
                current_message = {
                    "date": log['date'],
                    "time": log['time'],
                    "name": log['name'],
                    "message": log['message']
                }
            elif current_message['name'] == log['name']:
                current_message['message'] += '\n' + log['message']
            else:
                combined_messages.append(current_message)
                current_message = {
                    "date": log['date'],
                    "time": log['time'],
                    "name": log['name'],
                    "message": log['message']
                }

        if current_message is not None:
            combined_messages.append(current_message)

        return combined_messages
    
    def group_messages(self, parsed_logs: List[dict], hour_gap: int = 5) -> List[List[dict]]:
        """
        Groups the messages in parsed_logs into conversation segments whenever
        a gap of 'hour_gap' hours or more is found between consecutive messages.

        Parameters
        ----------
        parsed_logs : list of dict
            Each dictionary must have keys: 'date', 'time', 'name', 'message'.
            Example: {
                'date': '21/03/24',
                'time': '7:40 pm',
                'name': 'NNPY',
                'message': 'Hello'
            }

        hour_gap : int, optional
            The number of hours used to decide when a new conversation segment starts.
            Default is 5.

        Returns
        -------
        list of lists
            Each sub-list contains the messages (dictionaries) that belong to the same segment.
        """

        grouped = []
        current_group = []
        prev_dt = None

        for log in parsed_logs:
            # Clean up potential non-breaking spaces (e.g. \u202f) in the time
            time_str = log['time'].replace('\u202f', '').strip()

            # Build a datetime string from log['date'] and log['time']
            # Adjust strptime format to match your actual date/time format
            dt_str = f"{log['date']} {time_str}"
            
            # For a format like "21/03/24" + "7:40 pm", you could use:
            dt_format = "%d/%m/%y %I:%M%p"  # d/m/yy, 12-hour clock
            # If your data uses a 24-hour clock, you'd use '%d/%m/%y %H:%M'.
            # If your year is 2024 instead of '24', change to '%d/%m/%Y'.

            # Parse the datetime
            dt = datetime.strptime(dt_str, dt_format)

            if prev_dt is None:
                # First message, start the first group
                current_group = [log]
            else:
                # Calculate time difference in hours from the previous message
                diff_hours = (dt - prev_dt).total_seconds() / 3600.0
                if diff_hours < hour_gap:
                    # Still within the same conversation window
                    current_group.append(log)
                else:
                    # Time gap >= hour_gap, so we close off the old group and start a new one
                    grouped.append(current_group)
                    current_group = [log]

            prev_dt = dt

        # After the loop, if current_group has any messages, append it
        if current_group:
            grouped.append(current_group)

        return grouped

    def get_token_distribution(self, tokenizer: AutoTokenizer):
        """
        Calculate and visualize the token distribution of the parsed chat logs.

        Parameters
        ----------
        tokenizer : transformers.AutoTokenizer
            The tokenizer to use for counting tokens.

        Returns
        -------
        None

        Plots
        -----
        1. Histogram (Overall Token Distribution)
        2. Box Plot (By User)
        4. Time Series Analysis (Messages Over Time)
        """
        tokens_distribution = []
        global_msg_idx = 0
        for group in self.grouped_msgs:
            for msg in group:
                tokens = len(tokenizer(msg['message'])['input_ids'])
                tokens_distribution.append({
                    'global_idx': global_msg_idx,
                    'name': msg['name'],
                    'tokens': tokens
                })
                global_msg_idx += 1
        df = pd.DataFrame(tokens_distribution)
        # 1. Histogram (Overall Token Distribution)
        plt.figure(figsize=(12, 6))
        plt.hist(df['tokens'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Overall Token Distribution per Message')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.yscale('log')  # Use if you have long-tail distribution
        plt.show()

        # 2. Box Plot (By User)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='name', y='tokens', data=df)
        plt.title('Token Distribution by User')
        plt.xlabel('')
        plt.ylabel('Tokens per Message')
        plt.show()

        # 4. Time Series Analysis (Messages Over Time)
        # First convert timestamps to datetime objects
        all_messages = [msg for group in self.grouped_msgs for msg in group]
        timestamps = []
        for msg in all_messages:
            t = msg['time'].replace('\u202f', '')
            dt_str = f"{msg['date']} {t}"
            dt_format = "%d/%m/%y %I:%M%p"  # d/m/yy, 12-hour clock
            dt = datetime.strptime(dt_str, dt_format)
            timestamps.append(dt)

        # timestamps = [datetime.strptime(f"{msg['date']} {msg['time'].replace('\u202f', '')}", "%d/%m/%y %I:%M%p") for msg in all_messages]

        plt.figure(figsize=(16, 6))
        plt.scatter(timestamps, df['tokens'], alpha=0.5)
        plt.title('Token Counts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Tokens per Message')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_rc_msg_format(self, target_role: str):
        """
        Format the grouped messages into a format suitable for training a conversational
        model like DialoGPT or SFT. The format is a list of conversations, where each
        conversation is a list of turns. Each turn is a dictionary with two keys:
        'role' and 'content'. The 'role' can be either 'system' (for the conversation
        description), 'user', or 'assistant'. The 'content' is the message itself.

        Parameters
        ----------
        target_role : str
            The name of the user that should be treated as the target role for the
            conversation.

        Returns
        -------
        List[List[Dict[str, str]]]
            A list of conversations, where each conversation is a list of turns.
        """
        data = []
        for msg in self.grouped_msgs:
            grp = []
            grp.append(
                {
                    'role': 'system',
                    'content': "a conversation between interesting peoples"
                }
            )
            for turns in msg:
                grp.append(
                    {
                        'role': 'user' if target_role != turns['name'] else 'assistant',
                        'content': turns['message']
                    }
                )
            data.append(grp)

        return data


# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# processed_data = ProcessData("raw_chat.txt")
# processed_data.get_token_distribution(tokenizer)
# processed_data.get_rc_msg_format(target_role='Lil mo')
# print(processed_data.get_chat_names)
# print(processed_data.get_conversation_lengths)
# print(processed_data.grouped_msgs[-1])
