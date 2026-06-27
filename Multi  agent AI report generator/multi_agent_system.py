import os
import json
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

from colorama import init
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

# ======================================================
# SETUP
# ======================================================

init(autoreset=True)

console = Console()

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in .env file")

client = Groq(api_key=API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"

# ======================================================
# SHARED MEMORY
# ======================================================

class SharedMemory:

    def __init__(self):

        self.task_description = ""
        self.logs = []
        self.errors = []

    def log(self, agent, message, status="info"):

        self.logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "status": status,
            "message": message
        })

# ======================================================
# BASE AGENT
# ======================================================

class BaseAgent:

    def __init__(self, name, role, memory):

        self.name = name
        self.role = role
        self.memory = memory

    def ask_ai(self, system_prompt, user_prompt):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1024
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            error_msg = f"{self.name} Error: {str(e)}"

            self.memory.errors.append(error_msg)

            self.memory.log(
                self.name,
                error_msg,
                "error"
            )

            return f"ERROR: {str(e)}"

# ======================================================
# ORCHESTRATOR AGENT
# ======================================================

class OrchestratorAgent(BaseAgent):

    def __init__(self, memory):

        super().__init__(
            "Orchestrator",
            "Task Planner",
            memory
        )

    def run(self, task):

        self.memory.task_description = task

        self.memory.log(
            self.name,
            "Breaking task into subtasks",
            "start"
        )

        system_prompt = """
You are an expert project planner.

Break the task into 3-5 actionable subtasks.

Return ONLY JSON array.

Example:
["Task 1", "Task 2"]
"""

        user_prompt = f"Task:\n{task}"

        result = self.ask_ai(
            system_prompt,
            user_prompt
        )

        try:

            clean = result.strip()

            if clean.startswith("```json"):
                clean = clean[7:]

            if clean.startswith("```"):
                clean = clean[3:]

            if clean.endswith("```"):
                clean = clean[:-3]

            clean = clean.strip()

            subtasks = json.loads(clean)

        except Exception:

            subtasks = [
                line.strip("-•123456789. ")
                for line in result.splitlines()
                if line.strip()
            ]

        self.memory.log(
            self.name,
            f"Created {len(subtasks)} subtasks",
            "success"
        )

        return subtasks

# ======================================================
# RESEARCHER AGENT
# ======================================================

class ResearcherAgent(BaseAgent):

    def __init__(self, memory):

        super().__init__(
            "Researcher",
            "Information Gatherer",
            memory
        )

    def run(self, subtasks):

        results = {}

        self.memory.log(
            self.name,
            "Starting research",
            "start"
        )

        for subtask in subtasks:

            self.memory.log(
                self.name,
                f"Researching: {subtask}",
                "info"
            )

            system_prompt = """
You are a professional research analyst.

Provide:
- factual
- concise
- structured
- detailed information

Use bullet points when needed.
"""

            user_prompt = f"""
Main Task:
{self.memory.task_description}

Subtask:
{subtask}
"""

            result = self.ask_ai(
                system_prompt,
                user_prompt
            )

            results[subtask] = result

        self.memory.log(
            self.name,
            "Research completed",
            "success"
        )

        return results

# ======================================================
# ANALYST AGENT
# ======================================================

class AnalystAgent(BaseAgent):

    def __init__(self, memory):

        super().__init__(
            "Analyst",
            "Strategic Analyzer",
            memory
        )

    def run(self, research):

        self.memory.log(
            self.name,
            "Analyzing findings",
            "start"
        )

        combined = "\n\n".join(
            f"## {k}\n{v}"
            for k, v in research.items()
        )

        system_prompt = """
You are a senior strategic analyst.

Analyze the research and provide:
- patterns
- insights
- opportunities
- risks
- conclusions
- recommendations

Use structured sections.
"""

        user_prompt = combined

        analysis = self.ask_ai(
            system_prompt,
            user_prompt
        )

        self.memory.log(
            self.name,
            "Analysis completed",
            "success"
        )

        return analysis

# ======================================================
# WRITER AGENT
# ======================================================

class WriterAgent(BaseAgent):

    def __init__(self, memory):

        super().__init__(
            "Writer",
            "Technical Report Generator",
            memory
        )

    def run(self, analysis):

        self.memory.log(
            self.name,
            "Generating report",
            "start"
        )

        system_prompt = """
You are an expert technical writer.

Generate a professional report with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Recommendations
- Conclusion

Use proper Markdown formatting.
"""

        user_prompt = analysis

        report = self.ask_ai(
            system_prompt,
            user_prompt
        )

        self.memory.log(
            self.name,
            "Report generated",
            "success"
        )

        return report

# ======================================================
# MULTI AGENT SYSTEM
# ======================================================

class MultiAgentSystem:

    def __init__(self):

        self.memory = SharedMemory()

        self.orchestrator = OrchestratorAgent(self.memory)

        self.researcher = ResearcherAgent(self.memory)

        self.analyst = AnalystAgent(self.memory)

        self.writer = WriterAgent(self.memory)

    def print_logs(self):

        table = Table(
            title="📜 Agent Logs",
            border_style="cyan"
        )

        table.add_column("Time")
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Message")

        for log in self.memory.logs:

            table.add_row(
                log["time"],
                log["agent"],
                log["status"],
                log["message"]
            )

        console.print(table)

    def run(self, task):

        console.print(
            Panel.fit(
                "[bold cyan]🤖 Multi-Agent AI System[/bold cyan]\n"
                "[dim]Powered by Groq API[/dim]",
                border_style="cyan"
            )
        )

        console.print(
            f"\n[bold green]TASK:[/bold green] {task}\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:

            # ORCHESTRATOR
            p1 = progress.add_task(
                "[yellow]Planning tasks...",
                total=None
            )

            subtasks = self.orchestrator.run(task)

            progress.remove_task(p1)

            console.print("\n[bold yellow]Subtasks:[/bold yellow]")

            for i, subtask in enumerate(subtasks, 1):

                console.print(
                    f"{i}. {subtask}"
                )

            # RESEARCHER
            p2 = progress.add_task(
                "[cyan]Researching...",
                total=None
            )

            research = self.researcher.run(subtasks)

            progress.remove_task(p2)

            # ANALYST
            p3 = progress.add_task(
                "[magenta]Analyzing...",
                total=None
            )

            analysis = self.analyst.run(research)

            progress.remove_task(p3)

            # WRITER
            p4 = progress.add_task(
                "[blue]Writing report...",
                total=None
            )

            report = self.writer.run(analysis)

            progress.remove_task(p4)

        # FINAL REPORT

        console.print("\n" + "─" * 70)

        console.print(
            Panel(
                "[bold green]📄 FINAL REPORT[/bold green]",
                border_style="green"
            )
        )

        console.print(
            Markdown(report)
        )

        # LOGS

        console.print("\n" + "─" * 70)

        self.print_logs()

        # SAVE REPORT

        filename = "multi_agent_report.md"

        with open(filename, "w", encoding="utf-8") as f:

            f.write("# Multi-Agent AI Report\n\n")

            f.write(f"Task: {task}\n\n")

            f.write(report)

        console.print(
            f"\n[bold green]✔ Report saved:[/bold green] {filename}"
        )

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    system = MultiAgentSystem()

    console.print(
        Panel.fit(
            "[bold cyan]Enter your task below[/bold cyan]",
            border_style="cyan"
        )
    )

    task = input("\n>>> ")

    system.run(task)