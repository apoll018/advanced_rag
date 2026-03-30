
import time
from typing import List, Dict, Any

class PipelineSteps:
    """
    Rastreador de etapas do pipeline com timestamps e duração.
    """
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def add_step(self, name: str, description: str = ""):
        """Adiciona uma nova etapa ao pipeline."""
        self.steps.append({
            "name": name,
            "description": description,
            "status": "pending",  # pending, running, completed, error
            "start_time": None,
            "duration": None,
            "timestamp": None
        })
        return len(self.steps) - 1
    
    def start_step(self, index: int):
        """Marca uma etapa como iniciada."""
        if 0 <= index < len(self.steps):
            self.steps[index]["status"] = "running"
            self.steps[index]["start_time"] = time.time()
    
    def complete_step(self, index: int, duration: float = None):
        """Marca uma etapa como concluída."""
        if 0 <= index < len(self.steps):
            self.steps[index]["status"] = "completed"
            
            # ✅ CORRIGIDO: Garante que sempre calcula duração
            if duration is not None:
                self.steps[index]["duration"] = duration
            elif self.steps[index]["start_time"] is not None:
                self.steps[index]["duration"] = time.time() - self.steps[index]["start_time"]
            else:
                # Se não tiver start_time, assume 0
                self.steps[index]["duration"] = 0
                
            self.steps[index]["timestamp"] = time.time()
    
    def error_step(self, index: int, error_msg: str = ""):
        """Marca uma etapa como erro."""
        if 0 <= index < len(self.steps):
            self.steps[index]["status"] = "error"
            self.steps[index]["description"] = error_msg
            self.steps[index]["duration"] = time.time() - (self.steps[index]["start_time"] or self.start_time)
    
    def get_formatted_steps(self) -> str:
        """Retorna uma string formatada com todas as etapas."""
        output = ""
        total_duration = 0
        
        for i, step in enumerate(self.steps):
            icon = self._get_icon(step["status"])
            duration_str = self._format_duration(step["duration"]) if step["duration"] else "..."
            
            # ✅ USE O i AQUI
            output += f"{i+1}. {icon} **{step['name']}** `{duration_str}`\n"
            
            if step["description"]:
                output += f"   → {step['description']}\n"
            
            if step["duration"]:
                total_duration += step["duration"]
        
        output += f"\n⏱️ **Tempo Total: `{self._format_duration(total_duration)}`**"
        return output
    
    def get_steps_dict(self) -> List[Dict]:
        """Retorna os passos como dicionário (para JSON)."""
        return self.steps
    
    @staticmethod
    def _get_icon(status: str) -> str:
        """Retorna o ícone apropriado para cada status."""
        icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "error": "❌"
        }
        return icons.get(status, "❓")
    
    @staticmethod
    def _format_duration(duration: float) -> str:
        """Formata a duração em formato legível."""
        if duration is None:
            return "..."
        elif duration < 0.001:
            return f"{duration*1000000:.0f}µs"
        elif duration < 1:
            return f"{duration*1000:.2f}ms"
        elif duration < 60:
            return f"{duration:.2f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m {seconds:.2f}s"

    def summary(self) -> Dict[str, Any]:
        """Retorna um resumo das etapas."""
        completed = sum(1 for s in self.steps if s["status"] == "completed")
        failed = sum(1 for s in self.steps if s["status"] == "error")
        total_time = time.time() - self.start_time
        
        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "failed": failed,
            "total_duration": total_time,
            "success": failed == 0
        }