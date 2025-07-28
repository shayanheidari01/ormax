# example_full.py
import asyncio
import time
from ormax import Database, Model
from ormax.fields import *

# Example models for different databases
class User(Model):
    table_name = "users"
    id = IntegerField(primary_key=True, auto_increment=True)
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)

class Post(Model):
    table_name = "posts"
    id = IntegerField(primary_key=True, auto_increment=True)
    title = CharField(max_length=200)
    content = TextField()
    author_id = IntegerField(foreign_key='users.id')
    published = BooleanField(default=False)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)

async def test_database(db_url: str, db_name: str):
    """Test function for different databases"""
    print(f"\n=== Testing {db_name} ===")
    
    try:
        db = Database(db_url)
        await db.connect()
        print(f"✓ Connected to {db_name}")
        
        # Register models
        db.register_model(User)
        db.register_model(Post)
        
        # Clean start
        await db.drop_tables()
        print("✓ Tables dropped")
        
        # Create tables
        await db.create_tables()
        print("✓ Tables created")
        
        # Test single create
        user = await User.create(
            username="testuser",
            email="test@example.com"
        )
        print(f"✓ Created user: {user}")
        
        # Test bulk create
        start_time = time.time()
        users_data = []
        for i in range(100):
            users_data.append({
                'username': f'user_{i}',
                'email': f'user_{i}@example.com',
                'is_active': True
            })
        
        created_users = await User.bulk_create(users_data)
        bulk_time = time.time() - start_time
        print(f"✓ Created {len(created_users)} users in {bulk_time:.4f} seconds")
        
        # Test queries
        all_users = await User.objects().all()
        print(f"✓ Retrieved {len(all_users)} users")
        
        active_users = await User.objects().filter(is_active=True).count()
        print(f"✓ Counted {active_users} active users")
        
        # Test updates
        updated_count = await User.objects().filter(is_active=True).update(is_active=False)
        print(f"✓ Updated {updated_count} users")
        
        # Test deletes
        deleted_count = await User.objects().filter(is_active=False).delete()
        print(f"✓ Deleted {deleted_count} users")
        
        # Final count
        final_count = await User.objects().count()
        print(f"✓ Final user count: {final_count}")
        
        await db.disconnect()
        print(f"✓ Disconnected from {db_name}")
        
    except Exception as e:
        print(f"✗ Error with {db_name}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Test all supported databases"""
    
    # Test databases (uncomment the ones you want to test)
    databases = [
        # ("sqlite:///test.db", "SQLite"),
        # ("mysql://user:password@localhost/testdb", "MySQL"),
        # ("postgresql://user:password@localhost/testdb", "PostgreSQL"),
        # ("mssql://user:password@localhost/testdb", "Microsoft SQL Server"),
        # ("oracle://user:password@localhost:1521/XE", "Oracle"),
        # ("aurora://user:password@cluster-endpoint/testdb", "Amazon Aurora"),
    ]
    
    # For demo, we'll test with SQLite
    databases = [("sqlite:///full_example.db", "SQLite")]
    
    for db_url, db_name in databases:
        await test_database(db_url, db_name)

if __name__ == "__main__":
    asyncio.run(main())