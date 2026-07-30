import pytest
import pytest_asyncio
from ormax import Database, Model
from ormax.fields import AutoField, CharField, ForeignKeyField

class Author(Model):
    id = AutoField()
    name = CharField(max_length=100)

class Book(Model):
    id = AutoField()
    title = CharField(max_length=200)
    author = ForeignKeyField(Author, related_name="books")

@pytest_asyncio.fixture
async def db():
    database = Database("sqlite:///:memory:")
    await database.connect()
    database.register_model(Author, Book)
    await database.create_tables()
    yield database
    await database.disconnect()

@pytest.mark.asyncio
async def test_foreign_key_relationships(db):
    author = await Author.create(name="J.K. Rowling")
    book1 = await Book.create(title="Harry Potter 1", author=author)
    book2 = await Book.create(title="Harry Potter 2", author=author)

    # Reverse relationship queryset
    books = await author.books.all()
    assert len(books) == 2
    assert {b.title for b in books} == {"Harry Potter 1", "Harry Potter 2"}

    # Foreign Key Lookup across models
    rowling_books = await Book.objects().filter(author__name="J.K. Rowling").all()
    assert len(rowling_books) == 2

@pytest.mark.asyncio
async def test_prefetch_related(db):
    a1 = await Author.create(name="Author 1")
    a2 = await Author.create(name="Author 2")

    await Book.create(title="Book 1A", author=a1)
    await Book.create(title="Book 1B", author=a1)
    await Book.create(title="Book 2A", author=a2)

    # Prefetch reverse relation 'books' on Author
    authors = await Author.objects().prefetch_related("books").all()
    assert len(authors) == 2
    for author in authors:
        prefetched_books = getattr(author, "books", None)
        assert prefetched_books is not None

    # Prefetch forward relation 'author' on Book
    books = await Book.objects().prefetch_related("author").all()
    assert len(books) == 3
    for book in books:
        assert book.author is not None


class StringModel(Model):
    id = AutoField()
    name = CharField(max_length=100)

class StringRef(Model):
    id = AutoField()
    label = CharField(max_length=100)
    target = ForeignKeyField("StringModel")

class Category(Model):
    id = AutoField()
    name = CharField(max_length=100)
    parent = ForeignKeyField("self")


@pytest.mark.asyncio
async def test_string_target_and_self_referential_fk():
    database = Database("sqlite:///:memory:")
    await database.connect()
    database.register_model(StringModel, StringRef, Category)
    await database.create_tables()

    try:
        sm = await StringModel.create(name="Target A")
        sr = await StringRef.create(label="Ref 1", target=sm)

        refs = await StringRef.objects().filter(target__name="Target A").all()
        assert len(refs) == 1
        assert refs[0].label == "Ref 1"

        c1 = await Category.create(name="Parent Cat")
        c2 = await Category.create(name="Child Cat", parent=c1)

        children = await Category.objects().filter(parent__name="Parent Cat").all()
        assert len(children) == 1
        assert children[0].name == "Child Cat"
    finally:
        await database.disconnect()

