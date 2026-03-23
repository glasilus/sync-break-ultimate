"""Disc VPC 01 R-T — entry point."""
from rt_gui import RealtimeGUI

if __name__ == '__main__':
    print("=" * 60)
    print("Disc VPC 01 R-T")
    print("=" * 60)
    print("  F11  — Toggle fullscreen")
    print("  ESC  — Exit fullscreen")
    print("  F12  — Toggle second monitor")
    print("=" * 60)
    RealtimeGUI().mainloop()
