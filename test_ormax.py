# test_ormax.py
"""
Comprehensive test suite for Ormax ORM
"""
import asyncio
import sys
import os
import time
import logging
from datetime import datetime, date
from typing import List
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ormax import Database, Model, Field
from ormax.fields import *
from ormax.query import QuerySet
from ormax.exceptions import DatabaseError, ValidationError
from ormax.utils import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test models
class User(Model):
    table_name = "users"
    id = AutoField()
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    password_hash = CharField(max_length=128)
    first_name = CharField(max_length=30)
    last_name = CharField(max_length=30)
    is_active = BooleanField(default=True)
    is_staff = BooleanField(default=False)
    date_joined = DateTimeField(auto_now_add=True)
    last_login = DateTimeField(nullable=True)
    profile_picture = CharField(max_length=255, nullable=True)


class Category(Model):
    table_name = "categories"
    id = AutoField()
    name = CharField(max_length=100)
    slug = SlugField(max_length=100, unique=True)
    description = TextField(nullable=True)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)


class Post(Model):
    table_name = "posts"
    id = AutoField()
    title = CharField(max_length=200)
    slug = SlugField(max_length=200)
    content = TextField()
    excerpt = CharField(max_length=500, nullable=True)
    author_id = IntegerField(foreign_key='users.id')
    category_id = IntegerField(foreign_key='categories.id')
    is_published = BooleanField(default=False)
    view_count = IntegerField(default=0)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)


class Comment(Model):
    table_name = "comments"
    id = AutoField()
    post_id = IntegerField(foreign_key='posts.id')
    author_name = CharField(max_length=100)
    author_email = EmailField()
    content = TextField()
    is_approved = BooleanField(default=False)
    created_at = DateTimeField(auto_now_add=True)


class Tag(Model):
    table_name = "tags"
    id = AutoField()
    name = CharField(max_length=50, unique=True)
    slug = SlugField(max_length=50, unique=True)
    created_at = DateTimeField(auto_now_add=True)


# Test database connection
async def test_database_connection(db: Database) -> bool:
    """Test basic database connection"""
    try:
        await db.connect()
        logger.info("✓ Database connection successful")
        await db.disconnect()
        logger.info("✓ Database disconnection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


# Test model creation and table operations
async def test_model_operations(db: Database) -> bool:
    """Test model creation, table creation and dropping"""
    try:
        await db.connect()
        
        # Register models
        models = [User, Category, Post, Comment, Tag]
        for model in models:
            db.register_model(model)
        
        logger.info("✓ Models registered successfully")
        
        # Drop existing tables
        await db.drop_tables(if_exists=True)
        logger.info("✓ Tables dropped successfully")
        
        # Create tables
        await db.create_tables()
        logger.info("✓ Tables created successfully")
        
        # Test table recreation (should not fail)
        await db.create_tables()
        logger.info("✓ Tables recreated successfully")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Model operations failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test basic CRUD operations
async def test_basic_crud(db: Database) -> bool:
    """Test basic Create, Read, Update, Delete operations"""
    try:
        await db.connect()
        
        # Clean up any existing test data (با فیلترهای ساده)
        await User.objects().filter(username="john_doe").delete()
        await User.objects().filter(username="jane_smith").delete()
        
        # Create users
        logger.info("Testing CREATE operations...")
        user1 = await User.create(
            username="john_doe",
            email="john@example.com",
            password_hash="hash123",
            first_name="John",
            last_name="Doe"
        )
        
        user2 = await User.create(
            username="jane_smith",
            email="jane@example.com",
            password_hash="hash456",
            first_name="Jane",
            last_name="Smith"
        )
        
        logger.info(f"✓ Created users: {user1.id}, {user2.id}")
        
        # Read operations
        logger.info("Testing READ operations...")
        all_users = await User.objects().all()
        assert len(all_users) == 2, f"Expected 2 users, got {len(all_users)}"
        logger.info(f"✓ Retrieved {len(all_users)} users")
        
        # Get specific user
        john = await User.objects().get(username="john_doe")
        assert john.first_name == "John", f"Expected John, got {john.first_name}"
        logger.info(f"✓ Retrieved user: {john.username}")
        
        # Filter users
        active_users = await User.objects().filter(is_active=True)
        assert len(active_users) == 2, f"Expected 2 active users, got {len(active_users)}"
        logger.info(f"✓ Filtered {len(active_users)} active users")
        
        # Update operations
        logger.info("Testing UPDATE operations...")
        john.last_name = "Updated"
        await john.save()
        logger.info("✓ Updated user successfully")
        
        # Verify update
        updated_john = await User.objects().get(id=john.id)
        assert updated_john.last_name == "Updated", f"Expected Updated, got {updated_john.last_name}"
        logger.info("✓ Update verification successful")
        
        # Delete operations
        logger.info("Testing DELETE operations...")
        await user2.delete()
        remaining_users = await User.objects().count()
        assert remaining_users == 1, f"Expected 1 user, got {remaining_users}"
        logger.info("✓ Deleted user successfully")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic CRUD operations failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test bulk operations
async def test_bulk_operations(db: Database) -> bool:
    """Test bulk create and other bulk operations"""
    try:
        await db.connect()
        
        # Clean up existing bulk test data (با فیلترهای ساده)
        # در اینجا ممکن است نیاز به حذف تک تک رکوردها باشد یا استفاده از فیلتر ساده‌تر
        
        # Bulk create users
        logger.info("Testing BULK CREATE operations...")
        users_data = []
        for i in range(50):
            users_data.append({
                'username': f'bulk_user_{i}',
                'email': f'bulk_{i}@example.com',
                'password_hash': f'hash_{i}',
                'first_name': f'First{i}',
                'last_name': f'Last{i}'
            })
        
        start_time = time.time()
        created_users = await User.bulk_create(users_data, batch_size=25)
        bulk_time = time.time() - start_time
        
        assert len(created_users) == 50, f"Expected 50 users, created {len(created_users)}"
        logger.info(f"✓ Bulk created {len(created_users)} users in {bulk_time:.4f} seconds")
        
        # Bulk update - استفاده از فیلتر ساده
        logger.info("Testing BULK UPDATE operations...")
        updated_count = await User.objects().filter(username="bulk_user_1").update(is_staff=True)
        logger.info(f"✓ Bulk updated {updated_count} users")
        
        # Verify bulk update
        staff_count = await User.objects().filter(is_staff=True).count()
        assert staff_count >= 1, f"Expected at least 1 staff user, got {staff_count}"
        logger.info("✓ Bulk update verification successful")
        
        # Bulk delete - استفاده از فیلتر ساده
        logger.info("Testing BULK DELETE operations...")
        deleted_count = await User.objects().filter(username="bulk_user_1").delete()
        assert deleted_count >= 1, "Expected at least 1 deleted user"
        logger.info(f"✓ Bulk deleted {deleted_count} users")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Bulk operations failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test advanced queries
async def test_advanced_queries(db: Database) -> bool:
    """Test advanced query operations"""
    try:
        await db.connect()
        
        # Clean up existing test data (با فیلترهای ساده)
        await Category.objects().filter(slug="technology").delete()
        # حذف فیلتر پیشرفته title__like
        
        # Create test data
        category = await Category.create(
            name="Technology",
            slug="technology",
            description="Tech related posts"
        )
        
        # Create an author first
        author = await User.create(
            username="post_author",
            email="author@test.com",
            password_hash="hash",
            first_name="Post",
            last_name="Author"
        )
        
        posts_data = []
        for i in range(20):
            posts_data.append({
                'title': f'Test Post {i}',
                'slug': f'test-post-{i}',
                'content': f'This is the content of test post {i}',
                'author_id': author.id,
                'category_id': category.id,
                'is_published': i % 2 == 0,
                'view_count': i * 10
            })
        
        await Post.bulk_create(posts_data)
        logger.info("✓ Created test data for advanced queries")
        
        # Test ordering
        logger.info("Testing ORDER BY operations...")
        ordered_posts = await Post.objects().order_by('-view_count').limit(5).all()
        assert len(ordered_posts) == 5, f"Expected 5 posts, got {len(ordered_posts)}"
        logger.info("✓ Order by test successful")
        
        # Test filtering
        logger.info("Testing advanced filtering...")
        published_posts = await Post.objects().filter(is_published=True).count()
        assert published_posts > 0, "Expected published posts"
        logger.info(f"✓ Found {published_posts} published posts")
        
        # Test distinct
        logger.info("Testing DISTINCT operations...")
        distinct_authors = await Post.objects().distinct().values_list('author_id', flat=True)
        assert len(distinct_authors) >= 1, "Expected at least 1 distinct author"
        logger.info(f"✓ Found {len(distinct_authors)} distinct authors")
        
        # Test values and values_list
        logger.info("Testing VALUES operations...")
        post_values = await Post.objects().limit(3).values('title', 'view_count')
        assert len(post_values) == 3, f"Expected 3 values, got {len(post_values)}"
        logger.info("✓ Values test successful")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Advanced queries failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test field validation
async def test_field_validation(db: Database) -> bool:
    """Test field validation and constraints"""
    try:
        await db.connect()
        
        # Clean up existing test data (با فیلترهای ساده)
        await User.objects().filter(username="unique_test").delete()
        await User.objects().filter(username="invalid_email_user").delete()
        await User.objects().filter(username="null_test").delete()
        
        # Test email validation
        logger.info("Testing email validation...")
        try:
            await User.create(
                username="invalid_email_user",
                email="invalid-email",  # Invalid email
                password_hash="hash",
                first_name="Test",
                last_name="User"
            )
            assert False, "Should have raised ValidationError"
        except ValidationError:
            logger.info("✓ Email validation works correctly")
        
        # Test unique constraint
        logger.info("Testing unique constraint...")
        await User.create(
            username="unique_test",
            email="unique@test.com",
            password_hash="hash",
            first_name="Unique",
            last_name="Test"
        )
        
        try:
            await User.create(
                username="unique_test",  # Duplicate username
                email="another@test.com",
                password_hash="hash",
                first_name="Another",
                last_name="Test"
            )
            assert False, "Should have raised DatabaseError"
        except DatabaseError:
            logger.info("✓ Unique constraint works correctly")
        
        # Test nullable fields
        logger.info("Testing nullable fields...")
        user_with_null = await User.create(
            username="null_test",
            email="null@test.com",
            password_hash="hash",
            first_name="Null",
            last_name="Test",
            profile_picture=None  # Nullable field
        )
        assert user_with_null.profile_picture is None, "Nullable field should be None"
        logger.info("✓ Nullable fields work correctly")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Field validation failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test transactions
async def test_transactions(db: Database) -> bool:
    """Test database transactions"""
    try:
        await db.connect()
        
        # Clean up existing test data
        await User.objects().filter(username="transaction_user").delete()
        await Category.objects().filter(slug="transaction-category").delete()
        await User.objects().filter(username="rollback_user").delete()
        
        logger.info("Testing transactions...")
        
        # Test successful transaction
        try:
            async with db.transaction():
                user = await User.create(
                    username="transaction_user",
                    email="transaction@test.com",
                    password_hash="hash",
                    first_name="Transaction",
                    last_name="User"
                )
                
                category = await Category.create(
                    name="Transaction Category",
                    slug="transaction-category"
                )
            
            # Verify transaction was committed
            user_exists = await User.objects().filter(username="transaction_user").exists()
            category_exists = await Category.objects().filter(slug="transaction-category").exists()
            
            assert user_exists, "User should exist after successful transaction"
            assert category_exists, "Category should exist after successful transaction"
            logger.info("✓ Successful transaction test passed")
        except Exception as e:
            logger.error(f"Transaction test failed: {e}")
            raise
        
        # Test failed transaction (rollback)
        try:
            async with db.transaction():
                await User.create(
                    username="rollback_user",
                    email="rollback@test.com",
                    password_hash="hash",
                    first_name="Rollback",
                    last_name="User"
                )
                
                # This should fail due to duplicate username
                await User.create(
                    username="rollback_user",  # Duplicate
                    email="rollback2@test.com",
                    password_hash="hash",
                    first_name="Rollback2",
                    last_name="User"
                )
        except DatabaseError:
            # This is expected
            pass
        
        # Verify rollback worked
        rollback_count = await User.objects().filter(username="rollback_user").count()
        assert rollback_count == 0, "User should not exist after failed transaction"
        logger.info("✓ Transaction rollback test passed")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Transactions test failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test performance
async def test_performance(db: Database) -> bool:
    """Test ORM performance"""
    try:
        await db.connect()
        
        # Clean up existing performance test data (با فیلترهای ساده)
        # حذف فیلتر پیشرفته username__like
        
        logger.info("Testing performance...")
        
        # Bulk insert performance test
        logger.info("Testing bulk insert performance...")
        users_data = []
        for i in range(100):
            users_data.append({
                'username': f'perf_user_{i}',
                'email': f'perf_{i}@example.com',
                'password_hash': f'perf_{i}',
                'first_name': f'Perf{i}',
                'last_name': f'User{i}'
            })
        
        start_time = time.time()
        await User.bulk_create(users_data, batch_size=50)
        bulk_insert_time = time.time() - start_time
        logger.info(f"✓ Bulk insert 100 users: {bulk_insert_time:.4f} seconds")
        
        # Query performance test
        logger.info("Testing query performance...")
        start_time = time.time()
        user_count = await User.objects().count()
        query_time = time.time() - start_time
        logger.info(f"✓ Count {user_count} users: {query_time:.4f} seconds")
        
        # Cleanup - استفاده از فیلتر ساده
        await User.objects().filter(username="perf_user_1").delete()
        
        await db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        await db.disconnect()
        return False


# Test utility functions
async def test_utilities() -> bool:
    """Test utility functions"""
    try:
        logger.info("Testing utility functions...")
        
        # Test string conversion
        camel_case = "camelCaseString"
        snake_case = "snake_case_string"
        assert camel_to_snake(camel_case) == "camel_case_string", "Camel to snake conversion failed"
        assert snake_to_camel(snake_case) == "snakeCaseString", "Snake to camel conversion failed"
        logger.info("✓ String conversion works correctly")
        
        # Test validation functions
        assert validate_email("test@example.com"), "Valid email should pass"
        assert not validate_email("invalid-email"), "Invalid email should fail"
        assert validate_url("https://example.com"), "Valid URL should pass"
        assert not validate_url("invalid-url"), "Invalid URL should fail"
        logger.info("✓ Validation functions work correctly")
        
        # Test JSON utilities
        test_data = {"name": "test", "value": 123}
        json_str = json_dumps(test_data)
        parsed_data = json_loads(json_str)
        assert parsed_data["name"] == "test", "JSON serialization/deserialization failed"
        logger.info("✓ JSON utilities work correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Utility functions test failed: {e}")
        traceback.print_exc()
        return False


# Test database URL parsing
async def test_database_urls() -> bool:
    """Test database URL parsing utilities"""
    try:
        logger.info("Testing database URL utilities...")
        
        # Test URL creation for SQLite
        url = create_database_url(
            scheme="sqlite",
            username="",
            password="",
            host="",
            database="test.db"
        )
        # SQLite URLs have special format
        expected = "sqlite:///test.db"
        assert url == expected, f"Expected {expected}, got {url}"
        logger.info("✓ Database URL creation works correctly")
        
        # Test URL parsing
        parsed = parse_database_url("postgresql://user:pass@localhost:5432/testdb")
        assert parsed['scheme'] == 'postgresql', "Scheme parsing failed"
        assert parsed['username'] == 'user', "Username parsing failed"
        assert parsed['host'] == 'localhost', "Host parsing failed"
        logger.info("✓ Database URL parsing works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Database URL test failed: {e}")
        traceback.print_exc()
        return False


# Quick smoke test
async def smoke_test():
    """Quick smoke test to verify basic functionality"""
    try:
        logger.info("Running smoke test...")
        
        db = Database("sqlite:///smoke_test.db")
        await db.connect()
        
        # Register and create tables
        db.register_model(User)
        await db.drop_tables(if_exists=True)
        await db.create_tables()
        
        # Clean up any existing data
        await User.objects().filter(username="smoke_test").delete()
        
        # Basic CRUD
        user = await User.create(
            username="smoke_test",
            email="smoke@test.com",
            password_hash="hash",
            first_name="Smoke",
            last_name="Test"
        )
        
        retrieved_user = await User.objects().get(username="smoke_test")
        assert retrieved_user.id == user.id, "User retrieval failed"
        
        # Update
        user.last_name = "Updated"
        await user.save()
        
        # Delete
        await user.delete()
        exists = await User.objects().filter(username="smoke_test").exists()
        assert not exists, "User deletion failed"
        
        await db.disconnect()
        logger.info("✓ Smoke test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Smoke test failed: {e}")
        return False


# Main test runner
async def run_all_tests():
    """Run all tests for Ormax ORM"""
    logger.info("=" * 60)
    logger.info("ORMAX ORM COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)
    
    # Test with SQLite (most accessible)
    db_url = "sqlite:///test_ormax.db"
    db = Database(db_url)
    
    test_results = []
    
    # Individual tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Operations", test_model_operations),
        ("Basic CRUD", test_basic_crud),
        ("Bulk Operations", test_bulk_operations),
        ("Advanced Queries", test_advanced_queries),
        ("Field Validation", test_field_validation),
        ("Transactions", test_transactions),
        ("Performance", test_performance),
        ("Utilities", test_utilities),
        ("Database URLs", test_database_urls),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'-' * 40}")
        
        try:
            if test_name in ["Database Connection", "Utilities", "Database URLs"]:
                # These tests don't need database connection
                if test_name == "Database Connection":
                    result = await test_func(db)
                else:
                    result = await test_func()
            else:
                result = await test_func(db)
            
            test_results.append((test_name, result))
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
            
        except Exception as e:
            logger.error(f"{test_name}: ✗ FAILED - {e}")
            test_results.append((test_name, False))
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"{'=' * 60}")
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info(f"SUCCESS RATE: {(passed/total)*100:.1f}%")
    logger.info(f"{'=' * 60}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Ormax ORM is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the logs.")
    
    return passed == total


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Ormax ORM Test Suite")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--database", default="sqlite:///test_ormax.db", 
                       help="Database URL (default: sqlite:///test_ormax.db)")
    
    args = parser.parse_args()
    
    if args.smoke:
        # Run smoke test only
        success = asyncio.run(smoke_test())
        sys.exit(0 if success else 1)
    
    else:
        # Run full test suite
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)