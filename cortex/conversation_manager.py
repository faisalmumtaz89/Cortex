"""Conversation management for Cortex."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import uuid

class MessageRole(Enum):
    """Role of message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime
    tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'tokens': self.tokens,
            'metadata': self.metadata,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            tokens=data.get('tokens'),
            metadata=data.get('metadata', {}),
            message_id=data.get('message_id')
        )

@dataclass
class Conversation:
    """A conversation thread."""
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.conversation_id:
            self.conversation_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
    
    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            **kwargs
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_context(self, max_tokens: Optional[int] = None) -> List[Message]:
        """Get conversation context within token limit."""
        if max_tokens is None:
            return self.messages
        
        total_tokens = 0
        context = []
        
        for message in reversed(self.messages):
            message_tokens = message.tokens or len(message.content.split()) * 1.3
            if total_tokens + message_tokens > max_tokens:
                break
            context.insert(0, message)
            total_tokens += message_tokens
        
        return context
    
    def branch(self, from_message_id: str) -> 'Conversation':
        """Create a branch from a specific message."""
        branch_point = None
        for i, msg in enumerate(self.messages):
            if msg.message_id == from_message_id:
                branch_point = i
                break
        
        if branch_point is None:
            raise ValueError(f"Message {from_message_id} not found")
        
        return Conversation(
            conversation_id=str(uuid.uuid4()),
            title=f"{self.title} (branch)" if self.title else None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=self.messages[:branch_point + 1].copy(),
            metadata={'branched_from': self.conversation_id},
            parent_id=self.conversation_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'conversation_id': self.conversation_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary."""
        return cls(
            conversation_id=data['conversation_id'],
            title=data.get('title'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            messages=[Message.from_dict(msg) for msg in data['messages']],
            metadata=data.get('metadata', {}),
            parent_id=data.get('parent_id')
        )
    
    def to_markdown(self) -> str:
        """Export conversation to Markdown format."""
        lines = []
        
        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")
        
        lines.append(f"*Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"*Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        for message in self.messages:
            role_header = {
                MessageRole.SYSTEM: "### System",
                MessageRole.USER: "### User",
                MessageRole.ASSISTANT: "### Assistant"
            }[message.role]
            
            lines.append(role_header)
            lines.append("")
            lines.append(message.content)
            lines.append("")
        
        return "\n".join(lines)

class ConversationManager:
    """Manage conversations and persistence."""
    
    def __init__(self, config: Any):
        """Initialize conversation manager."""
        self.config = config
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation_id: Optional[str] = None
        
        self.save_dir = Path(config.conversation.save_directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if config.conversation.auto_save:
            self._init_database()
        
        self._load_recent_conversations()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistence."""
        self.db_path = self.save_dir / "conversations.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    parent_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY (parent_id) REFERENCES conversations(conversation_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    tokens INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
    
    def _load_recent_conversations(self) -> None:
        """Load recent conversations from storage."""
        if not hasattr(self, 'db_path') or not self.db_path.exists():
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT conversation_id, title, created_at, updated_at, parent_id, metadata
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (self.config.conversation.max_conversation_history,))
                
                for row in cursor:
                    conv_id = row[0]
                    
                    messages_cursor = conn.execute("""
                        SELECT role, content, timestamp, tokens, metadata, message_id
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY timestamp ASC
                    """, (conv_id,))
                    
                    messages = []
                    for msg_row in messages_cursor:
                        messages.append(Message(
                            role=MessageRole(msg_row[0]),
                            content=msg_row[1],
                            timestamp=datetime.fromisoformat(msg_row[2]),
                            tokens=msg_row[3],
                            metadata=json.loads(msg_row[4]) if msg_row[4] else {},
                            message_id=msg_row[5]
                        ))
                    
                    conversation = Conversation(
                        conversation_id=conv_id,
                        title=row[1],
                        created_at=datetime.fromisoformat(row[2]),
                        updated_at=datetime.fromisoformat(row[3]),
                        messages=messages,
                        metadata=json.loads(row[5]) if row[5] else {},
                        parent_id=row[4]
                    )
                    
                    self.conversations[conv_id] = conversation
                    
        except Exception as e:
            print(f"Warning: Failed to load conversations: {e}")
    
    def new_conversation(self, title: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            metadata={}
        )
        
        self.conversations[conversation.conversation_id] = conversation
        self.current_conversation_id = conversation.conversation_id
        
        if self.config.conversation.auto_save:
            self._save_conversation(conversation)
        
        return conversation
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current active conversation."""
        if self.current_conversation_id:
            return self.conversations.get(self.current_conversation_id)
        return None
    
    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation."""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            return True
        return False
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Message:
        """Add a message to a conversation."""
        conv_id = conversation_id or self.current_conversation_id
        
        if not conv_id:
            conversation = self.new_conversation()
            conv_id = conversation.conversation_id
        
        conversation = self.conversations.get(conv_id)
        if not conversation:
            raise ValueError(f"Conversation {conv_id} not found")
        
        message = conversation.add_message(role, content, **kwargs)
        
        if self.config.conversation.auto_save:
            self._save_message(conv_id, message)
        
        return message
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to database."""
        if not hasattr(self, 'db_path'):
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, title, created_at, updated_at, parent_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    conversation.conversation_id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    conversation.parent_id,
                    json.dumps(conversation.metadata)
                ))
        except Exception as e:
            print(f"Warning: Failed to save conversation: {e}")
    
    def _save_message(self, conversation_id: str, message: Message) -> None:
        """Save message to database."""
        if not hasattr(self, 'db_path'):
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO messages 
                    (message_id, conversation_id, role, content, timestamp, tokens, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    conversation_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    message.tokens,
                    json.dumps(message.metadata)
                ))
        except Exception as e:
            print(f"Warning: Failed to save message: {e}")
    
    def export_conversation(
        self,
        conversation_id: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Export conversation to specified format."""
        conv_id = conversation_id or self.current_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            raise ValueError("No conversation to export")
        
        conversation = self.conversations[conv_id]
        
        if format == "json":
            return json.dumps(conversation.to_dict(), indent=2)
        elif format == "markdown":
            return conversation.to_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_conversation(self, data: str, format: str = "json") -> Conversation:
        """Import conversation from string."""
        if format == "json":
            conv_dict = json.loads(data)
            conversation = Conversation.from_dict(conv_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.conversations[conversation.conversation_id] = conversation
        
        if self.config.conversation.auto_save:
            self._save_conversation(conversation)
            for message in conversation.messages:
                self._save_message(conversation.conversation_id, message)
        
        return conversation
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        return [
            {
                'id': conv.conversation_id,
                'title': conv.title or f"Conversation {conv.created_at.strftime('%Y-%m-%d %H:%M')}",
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat(),
                'message_count': len(conv.messages),
                'is_current': conv.conversation_id == self.current_conversation_id
            }
            for conv in sorted(
                self.conversations.values(),
                key=lambda x: x.updated_at,
                reverse=True
            )
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id not in self.conversations:
            return False
        
        del self.conversations[conversation_id]
        
        if self.current_conversation_id == conversation_id:
            self.current_conversation_id = None
        
        if hasattr(self, 'db_path'):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                    conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            except Exception as e:
                print(f"Warning: Failed to delete conversation from database: {e}")
        
        return True
    
    def clear_all(self) -> None:
        """Clear all conversations."""
        self.conversations.clear()
        self.current_conversation_id = None
        
        if hasattr(self, 'db_path'):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM messages")
                    conn.execute("DELETE FROM conversations")
            except Exception as e:
                print(f"Warning: Failed to clear database: {e}")