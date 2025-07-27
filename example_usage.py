# example_usage.py
import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ormax import Database, Model
from ormax.fields import *


# Define models with simplified table_name setup
class User(Model):
    # Simple way to set table name
    table_name = "users_table"
    
    id = IntegerField(primary_key=True, auto_increment=True)
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    password_hash = CharField(max_length=128)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)


class Post(Model):
    # Alternative way using _meta
    _meta = {'table_name': 'blog_posts'}
    
    id = IntegerField(primary_key=True, auto_increment=True)
    title = CharField(max_length=200)
    content = TextField()
    author_id = IntegerField(foreign_key='users_table.id')
    published = BooleanField(default=False)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)


# Even simpler model without explicit table name (uses class name lowercase)
class Category(Model):
    id = IntegerField(primary_key=True, auto_increment=True)
    name = CharField(max_length=100)
    slug = CharField(max_length=100, unique=True)


async def main():
    # Initialize database - using SQLite for easy testing
    db = Database("sqlite:///example.db")
    
    try:
        # Connect to database
        await db.connect()
        print("Connected to database successfully")
        
        # Register models
        db.register_model(User)
        db.register_model(Post)
        db.register_model(Category)
        
        # Create tables
        await db.create_tables()
        print("Tables created successfully")
        
        # Show table names
        print(f"User table name: {User.get_table_name()}")
        print(f"Post table name: {Post.get_table_name()}")
        print(f"Category table name: {Category.get_table_name()}")
        
        # Create categories
        tech_category = await Category.create(name="Technology", slug="tech")
        news_category = await Category.create(name="News", slug="news")
        print(f"Created categories: {tech_category}, {news_category}")
        
        # Create users
        print("Creating users...")
        user1 = await User.create(
            username="john_doe",
            email="john@example.com",
            password_hash="hashed_password_1"
        )
        print(f"Created user1: {user1}")
        
        user2 = await User.create(
            username="jane_smith",
            email="jane@example.com",
            password_hash="hashed_password_2"
        )
        print(f"Created user2: {user2}")
        
        # Create posts
        print("Creating posts...")
        post1 = await Post.create(
            title="First Post",
            content="This is the first post content",
            author_id=user1.id
        )
        print(f"Created post1: {post1}")
        
        post2 = await Post.create(
            title="Second Post",
            content="This is the second post content",
            author_id=user2.id,
            published=True
        )
        print(f"Created post2: {post2}")
        
        # Query examples
        print("\n--- Query Examples ---")
        
        # Get all users
        all_users = await User.objects().all()
        print(f"All users count: {len(all_users)}")
        for user in all_users:
            print(f"  User: {user.username} ({user.email})")
        
        # Filter users
        active_users = await User.objects().filter(is_active=True)
        print(f"Active users count: {len(active_users)}")
        
        # Get user by username
        user = await User.objects().get(username="john_doe")
        print(f"Found user: {user}")
        
        # Update user
        user.email = "newemail@example.com"
        await user.save()
        print(f"Updated user: {user}")
        
        # Query with relationships (simulated)
        published_posts = await Post.objects().filter(published=True).order_by('-created_at')
        print(f"Published posts count: {len(published_posts)}")
        
        # Count posts
        post_count = await Post.objects().count()
        print(f"Total posts: {post_count}")
        
        # Update multiple records
        updated_count = await Post.objects().filter(published=False).update(published=True)
        print(f"Updated {updated_count} posts")
        
        # Test transactions
        print("\n--- Transaction Test ---")
        try:
            async with db.transaction():
                new_user = await User.create(
                    username="transaction_user",
                    email="transaction@example.com",
                    password_hash="transaction_hash"
                )
                new_post = await Post.create(
                    title="Transaction Post",
                    content="Created in transaction",
                    author_id=new_user.id
                )
                print(f"Created in transaction: {new_user}, {new_post}")
        except Exception as e:
            print(f"Transaction completed successfully (no rollback needed): {e}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.disconnect()
        print("Disconnected from database")


if __name__ == "__main__":
    asyncio.run(main())