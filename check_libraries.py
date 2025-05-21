import importlib
import subprocess

def get_library_version(library_name):
    """Attempts to import a library and return its version."""
    try:
        module = importlib.import_module(library_name.replace('-', '_'))
        if hasattr(module, '__version__'):
            return module.__version__
        # For libraries like Flask where version is in flask.__version__
        # but module name might be different after replace('-', '_')
        # try importing with original name if different
        if library_name.count('-') > 0:
             module_original_name = importlib.import_module(library_name)
             if hasattr(module_original_name, '__version__'):
                return module_original_name.__version__
        return "Version not found"
    except ImportError:
        return None
    except Exception as e:
        return f"Error getting version: {e}"


def main():
    libraries = [
        "google-generative-ai",
        "sympy",
        "scipy",
        "pint",
        "matplotlib",
        "Flask",
        "requests",
    ]

    installed_libraries = {}
    missing_libraries = []
    requirements_content = ""

    print("Checking Python library installations...\n")

    for lib in libraries:
        version = get_library_version(lib)
        if version:
            print(f"SUCCESS: {lib} (Version: {version}) is installed.")
            installed_libraries[lib] = version
            requirements_content += f"{lib}=={version}\n"
        else:
            print(f"INFO: {lib} is NOT installed.")
            print(f"To install, run: pip install {lib}\n")
            missing_libraries.append(lib)
            requirements_content += f"{lib}\n"

    if missing_libraries:
        print("\n--- Some libraries are missing. ---")
        print("You can install them individually using the commands above,")
        print("or install all missing (or all listed) libraries from 'requirements.txt'.")

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("\nGenerated 'requirements.txt' with all specified libraries.")

    print("\n--- Instructions ---")
    print("1. To run this script:")
    print("   python check_libraries.py")
    print("\n2. To install all libraries listed in requirements.txt (recommended after running the script):")
    print("   pip install -r requirements.txt")
    print("\n3. To install only the missing libraries (if you prefer, after identifying them):")
    for lib in missing_libraries:
        print(f"   pip install {lib}")
    print("\n--- End of Check ---")

if __name__ == "__main__":
    main()
